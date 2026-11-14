# Tests of gemm for both micro kernels: column major, row major and general
# strides, sizes at the edges of tiles and blocks, several alpha and beta.
# Each element of C is compared with a reference sum with the error bound
#   |C - C_ref| <= 4 (k + 2) eps (|alpha| sum_l |a_il| |b_lj| + |beta| |c_ij|).
#
# For float the reference is computed in double, eps is that of float.
#
#   mojo run -I . test.mojo neon|sme [float]

from std.math import isnan
from std.utils.numerics import nan
from std.random import random_float64, seed
from std.sys import argv

from gemm import PtrT, Workspace, gemm
from ugemm_sme import fpsr_get, fpsr_set


def raw[dt: DType](mut A: List[Scalar[dt]]) -> PtrT[dt]:
    return A.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


# A matrix of m x n elements stored with the given layout in a buffer.
struct Strided[dt: DType](Movable):
    var data: List[Scalar[Self.dt]]
    var incRow: Int
    var incCol: Int

    # layout: 0 column major, 1 row major, 2 general strides
    def __init__(out self, m: Int, n: Int, layout: Int, fill_nan: Bool):
        if layout == 0:
            self.incRow = 1
            self.incCol = m
        elif layout == 1:
            self.incRow = n
            self.incCol = 1
        else:
            self.incRow = 3
            self.incCol = 3 * m + 1
        var len = (m - 1) * self.incRow + (n - 1) * self.incCol + 1
        self.data = List[Scalar[Self.dt]](length=max(len, 1), fill=0.0)
        for j in range(n):
            for i in range(m):
                self.data[i * self.incRow + j * self.incCol] = (
                    nan[Self.dt]() if fill_nan else Scalar[Self.dt](
                        random_float64(-1.0, 1.0)
                    )
                )

    def __getitem__(self, i: Int, j: Int) -> Float64:
        return Float64(self.data[i * self.incRow + j * self.incCol])


def check[SME: Bool, dt: DType](
    m: Int, n: Int, k: Int, alpha: Float64, beta: Float64, layout: Int,
    mut ws: Workspace[dt],
) -> Bool:
    var A = Strided[dt](m, k, layout, False)
    var B = Strided[dt](k, n, layout, False)
    # beta == 0: C must not be read
    var C = Strided[dt](m, n, layout, beta == 0.0)
    var C0 = Strided[dt](m, n, layout, False)
    for i in range(len(C0.data)):
        C0.data[i] = C.data[i]

    gemm[SME, dt](
        m, n, k, Scalar[dt](alpha), raw(A.data), A.incRow, A.incCol,
        raw(B.data), B.incRow, B.incCol, Scalar[dt](beta), raw(C.data),
        C.incRow, C.incCol, ws,
    )

    comptime eps = 2.220446049250313e-16 if dt == DType.float64 else (
        1.1920928955078125e-07
    )
    for j in range(n):
        for i in range(m):
            var s = 0.0
            var bound = 0.0
            for l in range(k):
                s += A[i, l] * B[l, j]
                bound += abs(A[i, l] * B[l, j])
            var cref = alpha * s
            bound = abs(alpha) * bound
            if beta != 0.0:
                cref += beta * C0[i, j]
                bound += abs(beta * C0[i, j])
            bound *= 4.0 * Float64(k + 2) * eps
            var cij = C[i, j]
            if isnan(cij) or abs(cij - cref) > bound:
                print("  WRONG: m n k =", m, n, k, " alpha beta =", alpha,
                      beta, " layout", layout, " at", i, j, ":", cij, cref)
                return False
    return True


def run[SME: Bool, dt: DType]() raises:
    seed(7)
    var ws = Workspace[dt]()
    var sizes = [1, 7, 31, 32, 33, 47, 129, 400, 1100]
    var scalars = [(1.0, 1.0), (1.0, 0.0), (2.5, -0.5), (-1.0, 1.0),
                   (0.0, 2.0), (0.0, 0.0)]
    var layouts = ["column major", "row major", "general strides"]
    var tests = 0
    var failed = 0
    for layout in range(3):
        for m in sizes:
            for n in sizes:
                for k in sizes:
                    # the large sizes (beyond MC, KC and NC) only in a few
                    # combinations, to keep the run short
                    var big = Int(m > 400) + Int(n > 400) + Int(k > 400)
                    if big > 1 or (big == 1 and m * n * k > 1100 * 129 * 33):
                        continue
                    for ab in scalars:
                        tests += 1
                        if not check[SME, dt](m, n, k, ab[0], ab[1], layout,
                                              ws):
                            failed += 1
        print(" ", layouts[layout], ": done")

    # FPSR: entering and leaving streaming mode sets all cumulative exception
    # flags; none may be left that cannot occur for finite random data
    # (invalid, division by zero, overflow, input denormal: mask 0x87;
    # inexact and underflow can be genuine for the NEON kernel)
    var before = fpsr_get()
    var ws2 = Workspace[dt]()
    var A = Strided[dt](64, 64, 0, False)
    var B = Strided[dt](64, 64, 0, False)
    var C = Strided[dt](64, 64, 0, False)
    fpsr_set(0)
    gemm[SME, dt](64, 64, 64, 1.0, raw(A.data), 1, 64, raw(B.data), 1, 64,
                  1.0, raw(C.data), 1, 64, ws2)
    var after = fpsr_get()
    fpsr_set(before)
    tests += 1
    if after & 0x87 != 0:
        print("  WRONG: FPSR exception flags after gemm:", hex(after))
        failed += 1

    print("tests:", tests, " failed:", failed)
    if failed != 0:
        raise Error("test failed")


def main() raises:
    var args = argv()
    if len(args) < 2 or (args[1] != "neon" and args[1] != "sme"):
        print("usage: test neon|sme [float]")
        return
    var single = len(args) > 2 and args[2] == "float"
    print("kernel:", args[1], "float" if single else "double")
    if args[1] == "sme":
        if single:
            run[True, DType.float32]()
        else:
            run[True, DType.float64]()
    else:
        if single:
            run[False, DType.float32]()
        else:
            run[False, DType.float64]()
