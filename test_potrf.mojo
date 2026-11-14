# Tests of potrf for both micro kernels of the Mojo BLAS and for the BLAS of
# Accelerate: lower and upper, column major, row major and general strides
# (Accelerate: falls back to the Mojo BLAS), several tile sizes and numbers
# of threads (0: chosen by potrf_select).
#
# A = M M^T / n + I with random M (symmetric positive definite).  Checked:
#   max |L L^T - A| <= 8 n eps max |A|   (lower triangle; upper: U^T U)
#   the other triangle is not touched,
#   info > 0 for a matrix that is not positive definite.
#
#   ./test-potrf neon|sme|accelerate|pk|pk3

from std.random import random_float64, seed
from std.sys import argv

from gemm import Ptr
from potrf import PotrfContext, potrf, potrf_pk, potrf_pk3


def raw(mut A: List[Float64]) -> Ptr:
    return A.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


# layout: 0 column major, 1 row major, 2 general strides
def strides(n: Int, layout: Int) -> Tuple[Int, Int]:
    if layout == 0:
        return (1, n)
    if layout == 1:
        return (n, 1)
    return (3, 3 * n + 1)


def check[SME: Bool, ACC: Int, PK: Int = 0](
    n: Int, lower: Bool, layout: Int, nb: Int, nthreads: Int
) raises -> Bool:
    var s = strides(n, layout)
    var r = s[0]
    var c = s[1]
    var len = (n - 1) * (r + c) + 1
    # random M, A = M M^T / n + I in a buffer with the given strides, the
    # unused elements of the buffer (and later the other triangle) are -7
    var M = List[Float64](length=n * n, fill=0.0)
    for i in range(n * n):
        M[i] = random_float64(-1.0, 1.0)
    var A = List[Float64](length=len, fill=-7.0)
    var amax = 0.0
    for i in range(n):
        for j in range(n):
            var x = 0.0
            for l in range(n):
                x += M[i * n + l] * M[j * n + l]
            x = x / Float64(n) + (1.0 if i == j else 0.0)
            A[i * r + j * c] = x
            amax = max(amax, abs(x))
    var A0 = A.copy()

    var info: Int
    comptime if PK == 1:
        var pkb = PotrfContext()
        info = potrf_pk(n, raw(A), r if lower else c, c if lower else r, pkb)
    elif PK == 3:
        var pkb = PotrfContext()
        info = potrf_pk3(n, raw(A), r if lower else c, c if lower else r, pkb)
    else:
        info = potrf[SME, ACC](lower, n, raw(A), r, c, nb, nthreads)
    if info != 0:
        print("  info =", info)
        return False

    # L(i, j) for i >= j: lower: A(i, j), upper: U(j, i) = A(j, i), i.e.
    # element (i, j) of L at A[i*rl + j*cl]
    var rl = r if lower else c
    var cl = c if lower else r

    comptime eps = 2.220446049250313e-16
    var err = 0.0
    for i in range(n):
        for j in range(i + 1):
            var x = 0.0
            for l in range(j + 1):
                x += A[i * rl + l * cl] * A[j * rl + l * cl]
            err = max(err, abs(x - A0[i * r + j * c]))
            # the other triangle (and the gaps of the buffer) unchanged
            if i != j:
                var o = j * r + i * c if lower else i * r + j * c
                if A[o] != A0[o]:
                    print("  other triangle changed at", i, j)
                    return False
    # elements of the buffer that are not elements of the matrix
    var inside = List[Bool](length=len, fill=False)
    for i in range(n):
        for j in range(n):
            inside[i * r + j * c] = True
    for p in range(len):
        if not inside[p] and A[p] != A0[p]:
            print("  gap of the buffer changed at", p)
            return False
    if err > 8.0 * Float64(n) * eps * amax:
        print("  error", err, "bound", 8.0 * Float64(n) * eps * amax)
        return False
    return True


# A symmetric matrix that is not positive definite: info must be the order
# of the first leading minor that is not positive definite.
def check_info[SME: Bool, ACC: Int, PK: Int = 0](n: Int, nb: Int, nthreads: Int) raises -> Bool:
    var A = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        A[i * n + i] = 1.0
    var bad = n * 2 // 3
    A[bad * n + bad] = -1.0
    var info: Int
    comptime if PK == 1:
        var pkb = PotrfContext()
        info = potrf_pk(n, raw(A), 1, n, pkb)
    elif PK == 3:
        var pkb = PotrfContext()
        info = potrf_pk3(n, raw(A), 1, n, pkb)
    else:
        info = potrf[SME, ACC](True, n, raw(A), 1, n, nb, nthreads)
    _ = A^  # A must live until here (potrf only got a raw pointer)
    if info != bad + 1:
        print("  info", info, "expected", bad + 1)
        return False
    return True


def run[SME: Bool, ACC: Int]() raises:
    seed(1)
    var sizes = [1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 300, 513]
    var tiles = [32, 48, 100, 256]
    var threads = [1, 4, 0]
    var tests = 0
    var failed = 0
    for n in sizes:
        for nb in tiles:
            for nthreads in threads:
                for layout in range(3):
                    for lower in [True, False]:
                        tests += 1
                        if not check[SME, ACC](n, lower, layout, nb,
                                                 nthreads):
                            failed += 1
                            print("FAILED: n", n, "nb", nb, "threads",
                                  nthreads, "layout", layout, "lower", lower)
    for n in [10, 100, 400]:
        for nb in tiles:
            for nthreads in threads:
                tests += 1
                if not check_info[SME, ACC](n, nb, nthreads):
                    failed += 1
                    print("FAILED info: n", n, "nb", nb, "threads", nthreads)
    print("tests:", tests, " failed:", failed)


# potrf_pk (pack once, small n): no tiles, no threads
def run_pk[PK: Int]() raises:
    seed(1)
    var sizes = [1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 300, 513]
    var tests = 0
    var failed = 0
    for n in sizes:
        for layout in range(3):
            for lower in [True, False]:
                tests += 1
                if not check[True, 0, PK](n, lower, layout, 0, 1):
                    failed += 1
                    print("FAILED: n", n, "layout", layout, "lower", lower)
    for n in [10, 100, 400]:
        tests += 1
        if not check_info[True, 0, PK](n, 0, 1):
            failed += 1
            print("FAILED info: n", n)
    print("tests:", tests, " failed:", failed)


def main() raises:
    var args = argv()
    var kernel = String(args[1]) if len(args) > 1 else String("neon")
    if kernel == "pk":
        print("potrf_pk (pack once), sme micro kernel")
        run_pk[1]()
    elif kernel == "pk3":
        print("potrf_pk3 (pack once, lookahead, persistent thread)")
        run_pk[3]()
    elif kernel == "accelerate":
        print("potrf, BLAS of Accelerate")
        run[True, 1]()
    elif kernel == "sme":
        print("potrf, Mojo BLAS, sme micro kernel")
        run[True, 0]()
    else:
        print("potrf, Mojo BLAS, neon micro kernel")
        run[False, 0]()
