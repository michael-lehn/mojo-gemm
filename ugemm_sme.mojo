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
# The kernel is inline assembly: Mojo 1.2.0 cannot mark a function as
# streaming (the function attribute arm_streaming is rejected), so streaming
# mode and ZA are entered and left inside the assembly block.
#
# The 32 x 16 tile of C is held transposed in the eight 8 x 8 tiles
# ZA0.D ... ZA7.D (SVL = 512 bits): tile ZA(i + 4j) holds rows 8i ... 8i+7
# of columns 8j ... 8j+7 of the tile of C, a horizontal slice r of ZA(i + 4j)
# is part of column 8j + r of C.  For each k: one 4-vector load of the
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
comptime KERNEL = """
    smstart
    ptrue p0.d
    ptrue pn8.d
    zero {za}
    mov x9, $1
    mov x10, $2
    mov x11, $0
    mov x15, #8
    mov x16, #16
    mov x17, #24
    cbz $6, 2f
    mov w12, #0
    mov x13, $3
1:
    add x14, x13, $5
    ld1d {za0h.d[w12, 0]}, p0/z, [x13]
    ld1d {za1h.d[w12, 0]}, p0/z, [x13, x15, lsl #3]
    ld1d {za2h.d[w12, 0]}, p0/z, [x13, x16, lsl #3]
    ld1d {za3h.d[w12, 0]}, p0/z, [x13, x17, lsl #3]
    ld1d {za4h.d[w12, 0]}, p0/z, [x14]
    ld1d {za5h.d[w12, 0]}, p0/z, [x14, x15, lsl #3]
    ld1d {za6h.d[w12, 0]}, p0/z, [x14, x16, lsl #3]
    ld1d {za7h.d[w12, 0]}, p0/z, [x14, x17, lsl #3]
    add x13, x13, $4
    add w12, w12, #1
    cmp w12, #8
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
    add x10, x10, #128
    subs x11, x11, #1
    b.ne 2b
    mov w12, #0
    mov x13, $3
3:
    add x14, x13, $5
    st1d {za0h.d[w12, 0]}, p0, [x13]
    st1d {za1h.d[w12, 0]}, p0, [x13, x15, lsl #3]
    st1d {za2h.d[w12, 0]}, p0, [x13, x16, lsl #3]
    st1d {za3h.d[w12, 0]}, p0, [x13, x17, lsl #3]
    st1d {za4h.d[w12, 0]}, p0, [x14]
    st1d {za5h.d[w12, 0]}, p0, [x14, x15, lsl #3]
    st1d {za6h.d[w12, 0]}, p0, [x14, x16, lsl #3]
    st1d {za7h.d[w12, 0]}, p0, [x14, x17, lsl #3]
    add x13, x13, $4
    add w12, w12, #1
    cmp w12, #8
    b.lt 3b
    smstop
"""


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
