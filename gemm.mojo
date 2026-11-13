# DGEMM in Mojo: C <- beta C + alpha A B
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
# Micro kernels: NEON (MR x NR = 8 x 6) in plain Mojo SIMD code below, SME2
# (32 x 16) in ugemm_sme.mojo.  The caller selects one: gemm[SME=False] or
# gemm[SME=True] (SME2 requires an Apple M4, M5 or another CPU with SME2,
# FEAT_SME_F64F64 and a streaming vector length of 512 bits).

from std.math import ceildiv

from ugemm_sme import ugemm_sme, MR_SME, NR_SME, fpsr_get, fpsr_set

comptime f64 = DType.float64
comptime Ptr = MutPointer[Float64, MutUntrackedOrigin]

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

comptime KUNROLL = 4  # unrolling of the loop over k in the NEON kernel


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


# A_ = mc x kc block of A in MR-row panels: panel p holds A(p*MR + i, l) at
# A_[p*MR*kc + l*MR + i]; the last panel is zero padded.
def pack_A[MR: Int](
    mc: Int, kc: Int, A: Ptr, incRowA: Int, incColA: Int, A_: Ptr
):
    var mp = ceildiv(mc, MR)
    for p in range(mp):
        var P = A_.unsafe_offset(p * MR * kc)
        var i0 = p * MR
        var mr = min(MR, mc - i0)
        if mr == MR and incRowA == 1:
            for l in range(kc):
                var a = A.unsafe_load[width=MR](i0 + l * incColA)
                P.unsafe_store(l * MR, a)
        else:
            for l in range(kc):
                for i in range(MR):
                    P[unsafe_offset=l * MR + i] = (
                        A[unsafe_offset=(i0 + i) * incRowA + l * incColA]
                        if i < mr else 0.0
                    )


# B_ = kc x nc block of B in NR-column panels: panel q holds B(l, q*NR + j)
# at B_[q*NR*kc + l*NR + j]; the last panel is zero padded.
def pack_B[NR: Int](
    kc: Int, nc: Int, B: Ptr, incRowB: Int, incColB: Int, B_: Ptr
):
    var np = ceildiv(nc, NR)
    for q in range(np):
        var P = B_.unsafe_offset(q * NR * kc)
        var j0 = q * NR
        var nr = min(NR, nc - j0)
        for l in range(kc):
            for j in range(NR):
                P[unsafe_offset=l * NR + j] = (
                    B[unsafe_offset=l * incRowB + (j0 + j) * incColB]
                    if j < nr else 0.0
                )


# ---------------------------------------------------------------------------
# NEON micro kernel: AB = A_panel * B_panel (MR x NR, in registers), then
# C <- beta C + alpha AB for the mr x nr part that exists.
# ---------------------------------------------------------------------------


# One rank-1 update of the tile: acc[j] += A(:, l) * B(l, j).
@always_inline
def rank1[MR: Int, NR: Int](
    l: Int, A: Ptr, B: Ptr, mut acc: Array[SIMD[f64, MR], NR]
):
    var a = A.unsafe_load[width=MR](l * MR)
    comptime for j in range(NR):
        # an FMA with a broadcast element of B
        acc[j] = a.fma(SIMD[f64, MR](B[unsafe_offset=l * NR + j]), acc[j])


def ugemm[MR: Int, NR: Int, KU: Int = KUNROLL](
    kc: Int, alpha: Float64, A: Ptr, B: Ptr, beta: Float64,
    mr: Int, nr: Int, C: Ptr, incRowC: Int, incColC: Int,
):
    # NR columns of the tile, each a SIMD vector of length MR
    var acc = Array[SIMD[f64, MR], NR](fill=SIMD[f64, MR](0.0))
    # the loop over k, unrolled KU times at compile time
    var l = 0
    while l + KU <= kc:
        comptime for u in range(KU):
            rank1[MR, NR](l + u, A, B, acc)
        l += KU
    while l < kc:
        rank1[MR, NR](l, A, B, acc)
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


def mgemm[MR: Int, NR: Int, SME: Bool](
    mc: Int, nc: Int, kc: Int, alpha: Float64, A_: Ptr, B_: Ptr,
    beta: Float64, C: Ptr, incRowC: Int, incColC: Int,
):
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
            comptime if SME:
                ugemm_sme(
                    kc, alpha, Ap, Bp, beta, mr, nr, Cp, incRowC, incColC
                )
            else:
                ugemm[MR, NR](
                    kc, alpha, Ap, Bp, beta, mr, nr, Cp, incRowC, incColC
                )
            ir += MR
        jr += NR


# ---------------------------------------------------------------------------
# Workspace and driver
# ---------------------------------------------------------------------------


# The packing buffers, owned by the caller and reused across calls (no
# allocation per call once they are large enough).
struct Workspace(Movable):
    var A_: List[Float64]
    var B_: List[Float64]

    def __init__(out self):
        self.A_ = List[Float64]()
        self.B_ = List[Float64]()

    def reserve(mut self, lenA: Int, lenB: Int):
        if len(self.A_) < lenA:
            self.A_ = List[Float64](length=lenA, fill=0.0)
        if len(self.B_) < lenB:
            self.B_ = List[Float64](length=lenB, fill=0.0)

    def ptr_A(mut self) -> Ptr:
        return self.A_.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    def ptr_B(mut self) -> Ptr:
        return self.B_.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


# C <- beta C (beta == 0: C is set to zero without being read)
def gescal(m: Int, n: Int, beta: Float64, C: Ptr, incRowC: Int,
           incColC: Int):
    if beta == 1.0:
        return
    for j in range(n):
        for i in range(m):
            var cij = C.unsafe_offset(i * incRowC + j * incColC)
            cij[] = 0.0 if beta == 0.0 else beta * cij[]


def gemm[SME: Bool](
    m: Int, n: Int, k: Int, alpha: Float64,
    A: Ptr, incRowA: Int, incColA: Int,
    B: Ptr, incRowB: Int, incColB: Int,
    beta: Float64, C: Ptr, incRowC: Int, incColC: Int,
    mut ws: Workspace,
):
    comptime c = SME2 if SME else NEON
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
        gemm[SME](
            n, m, k, alpha, B, incColB, incRowB, A, incColA, incRowA,
            beta, C, incColC, incRowC, ws,
        )
        return

    if m <= 0 or n <= 0:
        return
    if alpha == 0.0 or k <= 0:
        gescal(m, n, beta, C, incRowC, incColC)
        return

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
            var beta_ = beta if pc == 0 else 1.0
            pack_B[NR](
                kc, nc, B.unsafe_offset(pc * incRowB + jc * incColB),
                incRowB, incColB, B_,
            )
            var ic = 0
            while ic < m:
                var mc = min(MC, m - ic)
                pack_A[MR](
                    mc, kc, A.unsafe_offset(ic * incRowA + pc * incColA),
                    incRowA, incColA, A_,
                )
                mgemm[MR, NR, SME](
                    mc, nc, kc, alpha, A_, B_, beta_,
                    C.unsafe_offset(ic * incRowC + jc * incColC),
                    incRowC, incColC,
                )
                ic += MC
            pc += KC
        jc += NC

    comptime if SME:
        fpsr_set(fpsr)


