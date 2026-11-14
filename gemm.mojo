# GEMM in Mojo (double and float): C <- beta C + alpha A B
#
# The algorithm of BLIS and ulmBLAS:
#
#   for jc (NC columns of C and B)
#     for pc (KC):   pack B(pc:pc+kc, jc:jc+nc) into NR-column panels
#       for ic (MC): pack A(ic:ic+mc, pc:pc+kc) into MR-row panels
#         macro kernel: for each NR panel of B, each MR panel of A
#           micro kernel: MR x NR tile in registers, kc rank-1 updates
#
# Matrices are general strided: element (i, j) of A is A[i*incRowA +
# j*incColA], so column major (incRow = 1), row major (incCol = 1) and
# matrices with other constant strides are all supported.
#
# Micro kernels: NEON (MR x NR = 8 x 6 for double, 8 x 12 for float) in
# plain Mojo SIMD code below, SME2 (32 x 16 for double, 32 x 32 for float)
# in ugemm_sme.mojo.  The caller selects one: gemm[SME=False] or
# gemm[SME=True] (SME2 requires an Apple M4, M5 or another CPU with SME2,
# FEAT_SME_F64F64 and a streaming vector length of 512 bits).

from std.math import ceildiv

from ugemm_sme import (ugemm_sme, mgemm_sme, mgemm_sme32, MR_SME, NR_SME,
                       fpsr_get, fpsr_set, PtrF, pack32_transpose)

comptime f64 = DType.float64
comptime f32 = DType.float32
# pointer to elements of type dt; Ptr: double
comptime PtrT[dt: DType] = MutPointer[Scalar[dt], MutUntrackedOrigin]
comptime Ptr = PtrT[f64]

# Block sizes of ulmBLAS for double on an Apple M5 (ulmblas/config/blocksize.h)
@fieldwise_init
struct Config(ImplicitlyCopyable):
    var MC: Int
    var KC: Int
    var NC: Int
    var MR: Int
    var NR: Int


comptime NEON = Config(960, 1024, 4080, 8, 6)
comptime SME2 = Config(384, 1024, 1024, MR_SME, NR_SME)
# float (ulmblas/config/blocksize.h)
comptime NEON32 = Config(960, 1024, 4080, 8, 12)
comptime SME2_32 = Config(512, 2048, 1024, 32, 32)


def config[dt: DType, SME: Bool]() -> Config:
    comptime if dt == f32:
        return SME2_32 if SME else NEON32
    return SME2 if SME else NEON

comptime KUNROLL = 4  # unrolling of the loop over k in the NEON kernel
comptime PV = 8  # SIMD width for packing partial panels


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


# A_ = alpha times the mc x kc block of A in MR-row panels: panel p holds
# alpha*A(p*MR + i, l) at A_[p*MR*kc + l*MR + i]; the last panel is zero
# padded.  With alpha in the packed panels the micro kernel is called with
# alpha = 1, which is the case of the direct path of the SME2 kernel.
def pack_A[dt: DType, MR: Int](
    mc: Int, kc: Int, alpha: Scalar[dt], A: PtrT[dt], incRowA: Int,
    incColA: Int, A_: PtrT[dt],
):
    var mp = ceildiv(mc, MR)
    for p in range(mp):
        var P = A_.unsafe_offset(p * MR * kc)
        var i0 = p * MR
        var mr = min(MR, mc - i0)
        comptime if dt == f32 and MR == 32:
            # rows of A contiguous: transposed with ZA (SME2 kernel only)
            if mr == MR and incColA == 1 and incRowA != 1 and alpha == 1.0:
                pack32_transpose(
                    kc, rebind[PtrF](A.unsafe_offset(i0 * incRowA)),
                    incRowA, rebind[PtrF](P),
                )
                continue
        if mr == MR and incRowA == 1:
            for l in range(kc):
                var a = A.unsafe_load[width=MR](i0 + l * incColA)
                P.unsafe_store(l * MR, alpha * a)
        elif incRowA == 1:
            # partial panel, columns contiguous: vectors of PV, then the rest
            for l in range(kc):
                var a = A.unsafe_offset(i0 + l * incColA)
                var p = P.unsafe_offset(l * MR)
                var i = 0
                while i + PV <= mr:
                    p.unsafe_store(i, alpha * a.unsafe_load[width=PV](i))
                    i += PV
                while i < mr:
                    p[unsafe_offset=i] = alpha * a[unsafe_offset=i]
                    i += 1
                while i < MR:
                    p[unsafe_offset=i] = 0.0
                    i += 1
        else:
            for l in range(kc):
                for i in range(mr):
                    P[unsafe_offset=l * MR + i] = (
                        alpha * A[unsafe_offset=(i0 + i) * incRowA + l * incColA]
                    )
                for i in range(mr, MR):
                    P[unsafe_offset=l * MR + i] = 0.0


# B_ = kc x nc block of B in NR-column panels: panel q holds B(l, q*NR + j)
# at B_[q*NR*kc + l*NR + j]; the last panel is zero padded.
def pack_B[dt: DType, NR: Int](
    kc: Int, nc: Int, B: PtrT[dt], incRowB: Int, incColB: Int, B_: PtrT[dt]
):
    var np = ceildiv(nc, NR)
    for q in range(np):
        var P = B_.unsafe_offset(q * NR * kc)
        var j0 = q * NR
        var nr = min(NR, nc - j0)
        comptime if dt == f32 and NR == 32:
            # columns of B contiguous: transposed with ZA (SME2 kernel only)
            if nr == NR and incRowB == 1 and incColB != 1:
                pack32_transpose(
                    kc, rebind[PtrF](B.unsafe_offset(j0 * incColB)),
                    incColB, rebind[PtrF](P),
                )
                continue
        comptime if NR & (NR - 1) == 0:  # SIMD width: a power of two
            if nr == NR and incColB == 1:
                # rows of the panel are contiguous: SIMD loads
                for l in range(kc):
                    var b = B.unsafe_load[width=NR](l * incRowB + j0)
                    P.unsafe_store(l * NR, b)
                continue
        if incColB == 1:
            # partial panel, rows contiguous: vectors of PV, then the rest
            for l in range(kc):
                var b = B.unsafe_offset(l * incRowB + j0)
                var p = P.unsafe_offset(l * NR)
                var j = 0
                while j + PV <= nr:
                    p.unsafe_store(j, b.unsafe_load[width=PV](j))
                    j += PV
                while j < nr:
                    p[unsafe_offset=j] = b[unsafe_offset=j]
                    j += 1
                while j < NR:
                    p[unsafe_offset=j] = 0.0
                    j += 1
        else:
            # column by column (contiguous reads if incRowB = 1)
            for j in range(nr):
                var b = B.unsafe_offset((j0 + j) * incColB)
                for l in range(kc):
                    P[unsafe_offset=l * NR + j] = b[unsafe_offset=l * incRowB]
            for l in range(kc):
                for j in range(nr, NR):
                    P[unsafe_offset=l * NR + j] = 0.0


# ---------------------------------------------------------------------------
# NEON micro kernel: AB = A_panel * B_panel (MR x NR, in registers), then
# C <- beta C + alpha AB for the mr x nr part that exists.
# ---------------------------------------------------------------------------


# One rank-1 update of the tile: acc[j] += A(:, l) * B(l, j).
@always_inline
def rank1[dt: DType, MR: Int, NR: Int](
    l: Int, A: PtrT[dt], B: PtrT[dt], mut acc: Array[SIMD[dt, MR], NR]
):
    var a = A.unsafe_load[width=MR](l * MR)
    comptime for j in range(NR):
        # an FMA with a broadcast element of B
        acc[j] = a.fma(SIMD[dt, MR](B[unsafe_offset=l * NR + j]), acc[j])


def ugemm[dt: DType, MR: Int, NR: Int, KU: Int = KUNROLL](
    kc: Int, alpha: Scalar[dt], A: PtrT[dt], B: PtrT[dt], beta: Scalar[dt],
    mr: Int, nr: Int, C: PtrT[dt], incRowC: Int, incColC: Int,
):
    # NR columns of the tile, each a SIMD vector of length MR
    var acc = Array[SIMD[dt, MR], NR](fill=SIMD[dt, MR](0.0))
    # the loop over k, unrolled KU times at compile time
    var l = 0
    while l + KU <= kc:
        comptime for u in range(KU):
            rank1[dt, MR, NR](l + u, A, B, acc)
        l += KU
    while l < kc:
        rank1[dt, MR, NR](l, A, B, acc)
        l += 1

    if mr == MR and incRowC == 1:
        # full columns with unit row stride: SIMD loads and stores
        comptime for j in range(NR):
            if j < nr:
                var c = C.unsafe_offset(j * incColC)
                var v = alpha * acc[j]
                if beta != 0.0:
                    v = beta * c.unsafe_load[width=MR]() + v
                c.unsafe_store(0, v)
    else:
        comptime for j in range(NR):
            if j < nr:
                for i in range(mr):
                    var cij = C.unsafe_offset(i * incRowC + j * incColC)
                    var v = alpha * acc[j][i]
                    if beta != 0.0:
                        v += beta * cij[]
                    cij[] = v


# ---------------------------------------------------------------------------
# Macro kernel
# ---------------------------------------------------------------------------


def mgemm[dt: DType, MR: Int, NR: Int, SME: Bool](
    mc: Int, nc: Int, kc: Int, alpha: Scalar[dt], A_: PtrT[dt],
    B_: PtrT[dt], beta: Scalar[dt], C: PtrT[dt], incRowC: Int, incColC: Int,
):
    comptime if SME:
        # alpha is always 1: gemm puts alpha into A_
        comptime if dt == f32:
            mgemm_sme32(mc, nc, kc, rebind[PtrF](A_), rebind[PtrF](B_),
                        rebind[Float32](beta), rebind[PtrF](C), incRowC,
                        incColC)
            return
        else:
            if alpha == 1.0:
                mgemm_sme(mc, nc, kc, rebind[Ptr](A_), rebind[Ptr](B_),
                          rebind[Float64](beta), rebind[Ptr](C), incRowC,
                          incColC)
                return
    var jr = 0
    while jr < nc:
        var nr = min(NR, nc - jr)
        var ir = 0
        while ir < mc:
            var mr = min(MR, mc - ir)
            var Ap = A_.unsafe_offset(ir * kc)
            var Bp = B_.unsafe_offset(jr * kc)
            var Cp = C.unsafe_offset(ir * incRowC + jr * incColC)
            # the micro kernel is chosen at compile time
            comptime if SME and dt == f64:
                ugemm_sme(
                    kc, rebind[Float64](alpha), rebind[Ptr](Ap),
                    rebind[Ptr](Bp), rebind[Float64](beta), mr, nr,
                    rebind[Ptr](Cp), incRowC, incColC,
                )
            else:
                ugemm[dt, MR, NR](
                    kc, alpha, Ap, Bp, beta, mr, nr, Cp, incRowC, incColC
                )
            ir += MR
        jr += NR


# ---------------------------------------------------------------------------
# Workspace and driver
# ---------------------------------------------------------------------------


# The packing buffers, owned by the caller and reused across calls (no
# allocation per call once they are large enough).
struct Workspace[dt: DType = f64](Movable):
    var A_: List[Scalar[Self.dt]]
    var B_: List[Scalar[Self.dt]]

    def __init__(out self):
        self.A_ = List[Scalar[Self.dt]]()
        self.B_ = List[Scalar[Self.dt]]()

    def reserve(mut self, lenA: Int, lenB: Int):
        if len(self.A_) < lenA:
            self.A_ = List[Scalar[Self.dt]](length=lenA, fill=0.0)
        if len(self.B_) < lenB:
            self.B_ = List[Scalar[Self.dt]](length=lenB, fill=0.0)

    def ptr_A(mut self) -> PtrT[Self.dt]:
        return self.A_.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    def ptr_B(mut self) -> PtrT[Self.dt]:
        return self.B_.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


# C <- beta C (beta == 0: C is set to zero without being read)
def gescal[dt: DType](m: Int, n: Int, beta: Scalar[dt], C: PtrT[dt],
                     incRowC: Int, incColC: Int):
    if beta == 1.0:
        return
    for j in range(n):
        for i in range(m):
            var cij = C.unsafe_offset(i * incRowC + j * incColC)
            cij[] = 0.0 if beta == 0.0 else beta * cij[]


def gemm[SME: Bool, dt: DType = f64](
    m: Int, n: Int, k: Int, alpha: Scalar[dt],
    A: PtrT[dt], incRowA: Int, incColA: Int,
    B: PtrT[dt], incRowB: Int, incColB: Int,
    beta: Scalar[dt], C: PtrT[dt], incRowC: Int, incColC: Int,
    mut ws: Workspace[dt],
):
    comptime c = config[dt, SME]()
    comptime MC = c.MC
    comptime KC = c.KC
    comptime NC = c.NC
    comptime MR = c.MR
    comptime NR = c.NR
    comptime assert MC % MR == 0 and NC % NR == 0, "MC, NC: multiples of MR, NR"

    # The micro kernels write C fastest with incRowC = 1 (the SME2 kernel
    # stores the tile of C column by column).  If C is closer to row major,
    # compute C^T = B^T A^T instead, as in BLIS.
    if abs(incRowC) > abs(incColC):
        gemm[SME, dt](
            n, m, k, alpha, B, incColB, incRowB, A, incColA, incRowA,
            beta, C, incColC, incRowC, ws,
        )
        return

    if m <= 0 or n <= 0:
        return
    if alpha == 0.0 or k <= 0:
        gescal(m, n, beta, C, incRowC, incColC)
        return

    # The streaming macro kernel handles beta = 0 and beta = 1 in place: any
    # other beta is applied here, in ordinary mode.
    var beta0 = beta
    comptime if SME:
        if beta != 0.0 and beta != 1.0:
            gescal(m, n, beta, C, incRowC, incColC)
            beta0 = 1.0

    # packing buffers for the blocks of this problem
    var mc_max = min(MC, ceildiv(m, MR) * MR)
    var nc_max = min(NC, ceildiv(n, NR) * NR)
    var kc_max = min(KC, k)
    ws.reserve(mc_max * kc_max, kc_max * nc_max)
    var A_ = ws.ptr_A()
    var B_ = ws.ptr_B()

    # Entering and leaving streaming mode sets the cumulative exception flags
    # of FPSR; as in ulmBLAS, they are saved before and restored after.
    var fpsr = UInt64(0)
    comptime if SME:
        fpsr = fpsr_get()

    var jc = 0
    while jc < n:
        var nc = min(NC, n - jc)
        var pc = 0
        while pc < k:
            var kc = min(KC, k - pc)
            var beta_ = beta0 if pc == 0 else 1.0
            pack_B[dt, NR](
                kc, nc, B.unsafe_offset(pc * incRowB + jc * incColB),
                incRowB, incColB, B_,
            )
            var ic = 0
            while ic < m:
                var mc = min(MC, m - ic)
                pack_A[dt, MR](
                    mc, kc, alpha, A.unsafe_offset(ic * incRowA + pc * incColA),
                    incRowA, incColA, A_,
                )
                # alpha is in A_
                mgemm[dt, MR, NR, SME](
                    mc, nc, kc, 1.0, A_, B_, beta_,
                    C.unsafe_offset(ic * incRowC + jc * incColC),
                    incRowC, incColC,
                )
                ic += MC
            pc += KC
        jc += NC

    comptime if SME:
        fpsr_set(fpsr)


