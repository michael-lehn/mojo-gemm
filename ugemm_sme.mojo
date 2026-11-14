# SME2 micro kernel for double, MR x NR = 32 x 16.
#
# A Mojo port of the SME2 DGEMM micro kernel of ulmBLAS
# (https://github.com/michael-lehn/ulmBLAS, file
# ulmblas/level3/ukernel/sme/ugemm.tcc), whose structure is that of the
# double precision kernel bli_dgemm_armsme_int_4SVLx2SVL of BLIS pull
# request #921 (https://github.com/flame/blis/pull/921).
#
#   Copyright (C) 2014, The University of Texas at Austin
#   Copyright (C) 2021, The University of Tokyo
#   Copyright (c) 2014-2026, Michael C. Lehn and the ulmBLAS contributors
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#    - Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    - Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    - Neither the name(s) of the copyright holder(s) nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The kernel is inline assembly.  ugemm_sme enters and leaves streaming mode
# and ZA inside the assembly block (smstart/smstop), for every tile.
# mgemm_sme, the macro kernel, is a streaming function instead
# (@__llvm_metadata(`llvm.arm_streaming`); the caller switches the mode) and
# switches ZA on and off itself: once per call of the macro kernel, the
# assembly block of the micro kernel does not switch.
#
# The 32 x 16 tile of C is held transposed in the eight 8 x 8 tiles
# ZA0.D ... ZA7.D (SVL = 512 bits): tile ZA(i + 4j) holds rows 8i ... 8i+7
# of columns 8j ... 8j+7 of the tile of C, a horizontal slice r of ZA(i + 4j)
# is part of column 8j + r of C.  In the ZA array (rows of SVL bytes) the
# horizontal slice r of ZA(t) is row 8r + t, so column c of the tile is
# ZA rows 8c .. 8c+3 (c < 8) or 8(c-8)+4 .. 8(c-8)+7 (c >= 8): the full tile
# kernel moves C with LDR/STR (array vector), four rows per column, with
# the same immediate offset for the ZA row and the address.  For each k: one 4-vector load of the
# column of the panel of A, one 2-vector load of the row of the panel of B,
# eight FMOPA.

from std.sys._assembly import inlined_assembly
from std.sys.intrinsics import PrefetchOptions, prefetch

comptime Ptr = MutPointer[Float64, MutUntrackedOrigin]


# FPSR: entering and leaving streaming mode sets its cumulative exception
# flags (see ulmBLAS, ulmblas/level3/ukernel/sme/fpsr.h); the caller saves
# it before and restores it after the streaming routines.
@always_inline
def fpsr_get() -> UInt64:
    return inlined_assembly["mrs $0, fpsr", UInt64, constraints="=r"]()


@always_inline
def fpsr_set(value: UInt64):
    inlined_assembly["msr fpsr, $0", NoneType, constraints="r,~{memory}"](
        value
    )
comptime MR_SME = 32
comptime NR_SME = 16

# Streaming mode and ZA are entered and left inside the assembly block
# (smstart/smstop): every vector and predicate register is clobbered.
comptime CLOBBERS = (
    "~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17}"
    ",~{memory},~{cc}"
    ",~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"
    ",~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"
    ",~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23}"
    ",~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"
    ",~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7}"
    ",~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15}"
    ",~{ffr},~{fpmr}"
)

# Operands: $0 kc (>= 1), $1 A panel, $2 B panel, $3 C, $4 ldC in bytes,
# $5 8*ldC in bytes, $6 load C into ZA first (1) or start from zero (0).
# Result: C(0:32, 0:16) = (C if $6 else 0) + A_panel * B_panel.
comptime KERNEL_S_HEAD = """
    ptrue p0.d
    ptrue pn8.d
    zero {za}
    mov x9, $1
    mov x10, $2
    mov x11, $0
    cbz $6, 2f
    mov w12, #0
    mov x13, $3
1:
    add x14, x13, $5
    add w15, w12, #4
    ldr za[w12, 0], [x13]
    ldr za[w12, 1], [x13, #1, mul vl]
    ldr za[w12, 2], [x13, #2, mul vl]
    ldr za[w12, 3], [x13, #3, mul vl]
    ldr za[w15, 0], [x14]
    ldr za[w15, 1], [x14, #1, mul vl]
    ldr za[w15, 2], [x14, #2, mul vl]
    ldr za[w15, 3], [x14, #3, mul vl]
    add x13, x13, $4
    add w12, w12, #8
    cmp w12, #64
    b.lt 1b
2:
    ld1d {z0.d-z3.d}, pn8/z, [x9]
    ld1d {z4.d-z5.d}, pn8/z, [x10]
    fmopa za0.d, p0/m, p0/m, z4.d, z0.d
    fmopa za1.d, p0/m, p0/m, z4.d, z1.d
    fmopa za2.d, p0/m, p0/m, z4.d, z2.d
    fmopa za3.d, p0/m, p0/m, z4.d, z3.d
    fmopa za4.d, p0/m, p0/m, z5.d, z0.d
    fmopa za5.d, p0/m, p0/m, z5.d, z1.d
    fmopa za6.d, p0/m, p0/m, z5.d, z2.d
    fmopa za7.d, p0/m, p0/m, z5.d, z3.d
    add x9, x9, #256
"""
comptime KERNEL_S_TAIL = """    subs x11, x11, #1
    b.ne 2b
    mov w12, #0
    mov x13, $3
3:
    add x14, x13, $5
    add w15, w12, #4
    str za[w12, 0], [x13]
    str za[w12, 1], [x13, #1, mul vl]
    str za[w12, 2], [x13, #2, mul vl]
    str za[w12, 3], [x13, #3, mul vl]
    str za[w15, 0], [x14]
    str za[w15, 1], [x14, #1, mul vl]
    str za[w15, 2], [x14, #2, mul vl]
    str za[w15, 3], [x14, #3, mul vl]
    add x13, x13, $4
    add w12, w12, #8
    cmp w12, #64
    b.lt 3b
"""
comptime KERNEL_S = KERNEL_S_HEAD + "    add x10, x10, #128\n" + KERNEL_S_TAIL
# with the mode switches, for ordinary (non-streaming) callers
comptime KERNEL = "    smstart\n" + KERNEL_S + "    smstop\n"



# KERNEL_S: for streaming functions with ZA (no mode switches).
@always_inline
def kernel_32x16_s(kc: Int, A: Ptr, B: Ptr, C: Ptr, ldC: Int, load_c: Bool):
    inlined_assembly[
        KERNEL_S, NoneType, constraints="r,r,r,r,r,r,r," + CLOBBERS
    ](kc, A, B, C, ldC * 8, ldC * 64, Int(load_c))


# Partial tiles (mr <= 32 rows, nr <= 16 columns, incRowC = 1): as KERNEL_S,
# but C is loaded and stored with predicates, column by column.
# Operands: $0 kc (>= 1), $1 A panel, $2 B panel, $3 C, $4 ldC in bytes,
# $5 mr, $6 nr, $7 load C into ZA first (1) or start from zero (0).
comptime KERNEL_P = """
    ptrue p0.d
    ptrue pn8.d
    mov x15, #8
    mov x16, #16
    mov x17, #24
    whilelt p1.d, xzr, $5
    whilelt p2.d, x15, $5
    whilelt p3.d, x16, $5
    whilelt p4.d, x17, $5
    zero {za}
    mov x9, $1
    mov x10, $2
    mov x11, $0
    cbz $7, 4f
    mov x12, #0
    mov x13, $3
1:
    cmp x12, $6
    b.ge 4f
    cmp x12, #8
    b.ge 2f
    ld1d {za0h.d[w12, 0]}, p1/z, [x13]
    ld1d {za1h.d[w12, 0]}, p2/z, [x13, x15, lsl #3]
    ld1d {za2h.d[w12, 0]}, p3/z, [x13, x16, lsl #3]
    ld1d {za3h.d[w12, 0]}, p4/z, [x13, x17, lsl #3]
    b 3f
2:
    sub w14, w12, #8
    ld1d {za4h.d[w14, 0]}, p1/z, [x13]
    ld1d {za5h.d[w14, 0]}, p2/z, [x13, x15, lsl #3]
    ld1d {za6h.d[w14, 0]}, p3/z, [x13, x16, lsl #3]
    ld1d {za7h.d[w14, 0]}, p4/z, [x13, x17, lsl #3]
3:
    add x13, x13, $4
    add x12, x12, #1
    b 1b
4:
    ld1d {z0.d-z3.d}, pn8/z, [x9]
    ld1d {z4.d-z5.d}, pn8/z, [x10]
    fmopa za0.d, p0/m, p0/m, z4.d, z0.d
    fmopa za1.d, p0/m, p0/m, z4.d, z1.d
    fmopa za2.d, p0/m, p0/m, z4.d, z2.d
    fmopa za3.d, p0/m, p0/m, z4.d, z3.d
    fmopa za4.d, p0/m, p0/m, z5.d, z0.d
    fmopa za5.d, p0/m, p0/m, z5.d, z1.d
    fmopa za6.d, p0/m, p0/m, z5.d, z2.d
    fmopa za7.d, p0/m, p0/m, z5.d, z3.d
    add x9, x9, #256
    add x10, x10, #128
    subs x11, x11, #1
    b.ne 4b
    mov x12, #0
    mov x13, $3
5:
    cmp x12, $6
    b.ge 8f
    cmp x12, #8
    b.ge 6f
    st1d {za0h.d[w12, 0]}, p1, [x13]
    st1d {za1h.d[w12, 0]}, p2, [x13, x15, lsl #3]
    st1d {za2h.d[w12, 0]}, p3, [x13, x16, lsl #3]
    st1d {za3h.d[w12, 0]}, p4, [x13, x17, lsl #3]
    b 7f
6:
    sub w14, w12, #8
    st1d {za4h.d[w14, 0]}, p1, [x13]
    st1d {za5h.d[w14, 0]}, p2, [x13, x15, lsl #3]
    st1d {za6h.d[w14, 0]}, p3, [x13, x16, lsl #3]
    st1d {za7h.d[w14, 0]}, p4, [x13, x17, lsl #3]
7:
    add x13, x13, $4
    add x12, x12, #1
    b 5b
8:
"""


@always_inline
def kernel_partial_s(kc: Int, A: Ptr, B: Ptr, C: Ptr, ldC: Int, mr: Int,
                     nr: Int, load_c: Bool):
    inlined_assembly[
        KERNEL_P, NoneType, constraints="r,r,r,r,r,r,r,r," + CLOBBERS
    ](kc, A, B, C, ldC * 8, mr, nr, Int(load_c))


@always_inline
def kernel_32x16(kc: Int, A: Ptr, B: Ptr, C: Ptr, ldC: Int, load_c: Bool):
    inlined_assembly[
        KERNEL, NoneType, constraints="r,r,r,r,r,r,r," + CLOBBERS
    ](kc, A, B, C, ldC * 8, ldC * 64, Int(load_c))


# The interface of the Mojo micro kernels (see ugemm in gemm.mojo):
# C(0:mr, 0:nr) <- beta C + alpha A_panel B_panel.
def ugemm_sme(
    kc: Int, alpha: Float64, A: Ptr, B: Ptr, beta: Float64,
    mr: Int, nr: Int, C: Ptr, incRowC: Int, incColC: Int,
):
    if (alpha == 1.0 and (beta == 0.0 or beta == 1.0) and mr == MR_SME
            and nr == NR_SME and incRowC == 1):
        # direct: C is loaded into ZA (beta == 1) and stored from ZA.
        # As in ulmBLAS: prefetch the tile of C below (the tile of the next
        # call of the macro kernel) into L2, in ordinary mode, before the
        # kernel enters streaming mode.  PRFM does not fault outside of C.
        comptime pf = PrefetchOptions().for_write().medium_locality()
        comptime for c in range(NR_SME):
            var Cc = C.unsafe_offset(c * incColC)
            prefetch[pf](Cc.unsafe_offset(32))
            prefetch[pf](Cc.unsafe_offset(48))
            prefetch[pf](Cc.unsafe_offset(63))
        kernel_32x16(kc, A, B, C, incColC, beta == 1.0)
        return
    # otherwise: A_panel B_panel into a local column-major tile, then update
    var T = Array[Float64, MR_SME * NR_SME](uninitialized=True)
    var T_ = T.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    kernel_32x16(kc, A, B, T_, MR_SME, False)
    for j in range(nr):
        for i in range(mr):
            var cij = C.unsafe_offset(i * incRowC + j * incColC)
            var v = alpha * T_[unsafe_offset=i + j * MR_SME]
            if beta != 0.0:
                v += beta * cij[]
            cij[] = v


# Macro kernel in streaming mode: C(0:mc, 0:nc) <- beta C + A_ B_ for the
# packed blocks A_ (MR-row panels) and B_ (NR-column panels), see mgemm in
# gemm.mojo.  With incRowC = 1 and beta 0 or 1 (the case gemm arranges) the
# tiles are computed in place, partial ones with predicates; otherwise
# through a local tile.
@__llvm_metadata(`llvm.arm_streaming`)
@no_inline
def mgemm_sme(
    mc: Int, nc: Int, kc: Int, A_: Ptr, B_: Ptr, beta: Float64, C: Ptr,
    incRowC: Int, incColC: Int,
):
    var T = Array[Float64, MR_SME * NR_SME](uninitialized=True)
    var T_ = T.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var direct_ok = incRowC == 1 and (beta == 0.0 or beta == 1.0)
    # llvm.arm_new_za has no effect in Mojo 1.2.0: ZA is switched on here
    inlined_assembly["smstart za", NoneType, constraints="~{memory}"]()
    comptime pf = PrefetchOptions().for_write().medium_locality()
    var jr = 0
    while jr < nc:
        var nr = min(NR_SME, nc - jr)
        var ir = 0
        while ir < mc:
            var mr = min(MR_SME, mc - ir)
            var Ap = A_.unsafe_offset(ir * kc)
            var Bp = B_.unsafe_offset(jr * kc)
            var Cp = C.unsafe_offset(ir * incRowC + jr * incColC)
            if direct_ok and mr == MR_SME and nr == NR_SME:
                # As in ulmBLAS: prefetch the tile of C below (the tile of
                # the next call of the macro kernel) into L2.  PRFM is slow
                # in streaming mode (about 65 ns for these 48 on the M5),
                # but without it large matrices lose about 4%.
                comptime for c in range(NR_SME):
                    var Cc = Cp.unsafe_offset(c * incColC)
                    prefetch[pf](Cc.unsafe_offset(32))
                    prefetch[pf](Cc.unsafe_offset(48))
                    prefetch[pf](Cc.unsafe_offset(63))
                kernel_32x16_s(kc, Ap, Bp, Cp, incColC, beta == 1.0)
            elif direct_ok:
                kernel_partial_s(kc, Ap, Bp, Cp, incColC, mr, nr, beta == 1.0)
            else:
                kernel_32x16_s(kc, Ap, Bp, T_, MR_SME, False)
                for j in range(nr):
                    for i in range(mr):
                        var cij = Cp.unsafe_offset(i * incRowC + j * incColC)
                        var v = T_[unsafe_offset=i + j * MR_SME]
                        if beta != 0.0:
                            v += beta * cij[]
                        cij[] = v
            ir += MR_SME
        jr += NR_SME
    inlined_assembly["smstop za", NoneType, constraints="~{memory}"]()



# ---------------------------------------------------------------------------
# Kernels of potrf_pk (potrf.mojo): all operands in panels of 32 rows,
# column by column (32 doubles per column), the C tile 32 x 16 with leading
# dimension 32.  For streaming functions with ZA.
# ---------------------------------------------------------------------------

# $0 kc (>= 1), $1 A, $2 B (16 rows of a panel), $3 C in, $4 C out:
# C out = C in - A B^T (FMOPS)
comptime KERNEL_PK_UPD = """
    ptrue p0.d
    ptrue pn8.d
    mov x9, $1
    mov x10, $2
    mov x11, $0
    mov w12, #0
    mov x13, $3
1:
    add x14, x13, #2048
    add w15, w12, #4
    ldr za[w12, 0], [x13]
    ldr za[w12, 1], [x13, #1, mul vl]
    ldr za[w12, 2], [x13, #2, mul vl]
    ldr za[w12, 3], [x13, #3, mul vl]
    ldr za[w15, 0], [x14]
    ldr za[w15, 1], [x14, #1, mul vl]
    ldr za[w15, 2], [x14, #2, mul vl]
    ldr za[w15, 3], [x14, #3, mul vl]
    add x13, x13, #256
    add w12, w12, #8
    cmp w12, #64
    b.lt 1b
2:
    ld1d {z0.d-z3.d}, pn8/z, [x9]
    ld1d {z4.d-z5.d}, pn8/z, [x10]
    fmops za0.d, p0/m, p0/m, z4.d, z0.d
    fmops za1.d, p0/m, p0/m, z4.d, z1.d
    fmops za2.d, p0/m, p0/m, z4.d, z2.d
    fmops za3.d, p0/m, p0/m, z4.d, z3.d
    fmops za4.d, p0/m, p0/m, z5.d, z0.d
    fmops za5.d, p0/m, p0/m, z5.d, z1.d
    fmops za6.d, p0/m, p0/m, z5.d, z2.d
    fmops za7.d, p0/m, p0/m, z5.d, z3.d
    add x9, x9, #256
    add x10, x10, #256
    subs x11, x11, #1
    b.ne 2b
    mov w12, #0
    mov x13, $4
3:
    add x14, x13, #2048
    add w15, w12, #4
    str za[w12, 0], [x13]
    str za[w12, 1], [x13, #1, mul vl]
    str za[w12, 2], [x13, #2, mul vl]
    str za[w12, 3], [x13, #3, mul vl]
    str za[w15, 0], [x14]
    str za[w15, 1], [x14, #1, mul vl]
    str za[w15, 2], [x14, #2, mul vl]
    str za[w15, 3], [x14, #3, mul vl]
    add x13, x13, #256
    add w12, w12, #8
    cmp w12, #64
    b.lt 3b
"""

# $0 kc (>= 1), $1 A, $2 B (16 rows of a panel), $3 C: C = A B^T
comptime KERNEL_PK_SLV = """
    ptrue p0.d
    ptrue pn8.d
    mov x9, $1
    mov x10, $2
    mov x11, $0
    zero {za}
2:
    ld1d {z0.d-z3.d}, pn8/z, [x9]
    ld1d {z4.d-z5.d}, pn8/z, [x10]
    fmopa za0.d, p0/m, p0/m, z4.d, z0.d
    fmopa za1.d, p0/m, p0/m, z4.d, z1.d
    fmopa za2.d, p0/m, p0/m, z4.d, z2.d
    fmopa za3.d, p0/m, p0/m, z4.d, z3.d
    fmopa za4.d, p0/m, p0/m, z5.d, z0.d
    fmopa za5.d, p0/m, p0/m, z5.d, z1.d
    fmopa za6.d, p0/m, p0/m, z5.d, z2.d
    fmopa za7.d, p0/m, p0/m, z5.d, z3.d
    add x9, x9, #256
    add x10, x10, #256
    subs x11, x11, #1
    b.ne 2b
    mov w12, #0
    mov x13, $3
3:
    add x14, x13, #2048
    add w15, w12, #4
    str za[w12, 0], [x13]
    str za[w12, 1], [x13, #1, mul vl]
    str za[w12, 2], [x13, #2, mul vl]
    str za[w12, 3], [x13, #3, mul vl]
    str za[w15, 0], [x14]
    str za[w15, 1], [x14, #1, mul vl]
    str za[w15, 2], [x14, #2, mul vl]
    str za[w15, 3], [x14, #3, mul vl]
    add x13, x13, #256
    add w12, w12, #8
    cmp w12, #64
    b.lt 3b
"""


@always_inline
def kernel_pk_upd(kc: Int, A: Ptr, B: Ptr, Cin: Ptr, Cout: Ptr):
    inlined_assembly[
        KERNEL_PK_UPD, NoneType, constraints="r,r,r,r,r," + CLOBBERS
    ](kc, A, B, Cin, Cout)


@always_inline
def kernel_pk_slv(kc: Int, A: Ptr, B: Ptr, C: Ptr):
    inlined_assembly[
        KERNEL_PK_SLV, NoneType, constraints="r,r,r,r," + CLOBBERS
    ](kc, A, B, C)


# Panel storage of potrf_pk: panel P holds rows 32P .. 32P+31 of L for the
# columns 0 .. 32P+31, column by column (32 doubles each), panels one after
# the other.
@always_inline
def pk_offset(P: Int) -> Int:
    return 1024 * P * (P + 1) // 2


# Update of block column K (K >= 1) for the panels P = K .. np-1 (from P0
# on if P0 >= 0):
# T(P) = C(P, K) - L(P, 0:32K) L(K, 0:32K)^T, T(P) at T + (P-K)*1024.  Both
# operands are the panels themselves (no packing).
@__llvm_metadata(`llvm.arm_streaming`)
@no_inline
def pk_update(buf: Ptr, T: Ptr, K: Int, np: Int, P0: Int = -1):
    inlined_assembly["smstart za", NoneType, constraints="~{memory}"]()
    var kc = 32 * K
    var Bk = buf.unsafe_offset(pk_offset(K))
    for P in range(K if P0 < 0 else P0, np):
        var Ap = buf.unsafe_offset(pk_offset(P))
        var C = Ap.unsafe_offset(K * 1024)
        var Tp = T.unsafe_offset((P - K) * 1024)
        kernel_pk_upd(kc, Ap, Bk, C, Tp)
        kernel_pk_upd(kc, Ap, Bk.unsafe_offset(16), C.unsafe_offset(512),
                      Tp.unsafe_offset(512))
    inlined_assembly["smstop za", NoneType, constraints="~{memory}"]()


# Solve below the diagonal block of block column K: C(P, K) = T(P) Linv^T
# for P = K+1 .. np-1 (from P0 on if P0 >= 0), Linv the inverse of the
# diagonal block (column major,
# leading dimension 32).
@__llvm_metadata(`llvm.arm_streaming`)
@no_inline
def pk_solve(buf: Ptr, T: Ptr, Linv: Ptr, K: Int, np: Int, P0: Int = -1):
    inlined_assembly["smstart za", NoneType, constraints="~{memory}"]()
    for P in range(K + 1 if P0 < 0 else P0, np):
        var C = buf.unsafe_offset(pk_offset(P) + K * 1024)
        var Tp = T.unsafe_offset((P - K) * 1024)
        kernel_pk_slv(32, Tp, Linv, C)
        kernel_pk_slv(32, Tp, Linv.unsafe_offset(16), C.unsafe_offset(512))
    inlined_assembly["smstop za", NoneType, constraints="~{memory}"]()
