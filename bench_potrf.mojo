# Benchmark of DPOTRF: Cholesky factorization A = L L^T (or U^T U).
#
#   ./bench-potrf IMPL [--uplo=L|U] [--order=row|col] [--threads=T]
#                      [--nb=NB] [n ...]
#
#   IMPL   neon        Mojo potrf, Mojo BLAS with NEON micro kernel
#          sme         Mojo potrf, Mojo BLAS with SME2 micro kernel
#          accelerate-blas
#                      Mojo potrf, BLAS of Accelerate
#          PATH        dpotrf of the shared library PATH (LAPACK interface,
#                      column major; Accelerate: dpotrf$NEWLAPACK)
#   --threads, --nb    threads and tile size of the Mojo potrf;
#                      --threads=auto: both from potrf_select (tuning)
#   defaults: lower, column major, 1 thread, nb = 256,
#             n = 50, 100, ..., 4000
#
# A = M M^T / n + I with random M.  Per n one checked call (400 random
# elements of L L^T against A, error bound 8 n eps max |A|), which is also
# the warm-up, extended to at least 0.3 s; then 3 samples of at least 0.2 s
# each, the median is reported.  Before every call A is restored (not
# timed).  GFLOP/s from n^3 / 3 operations.

from std.ffi import OwnedDLHandle, external_call
from std.random import random_float64, random_ui64, seed
from std.sys import argv
from std.time import perf_counter_ns

from gemm import Ptr, Workspace, gemm
from potrf import PotrfContext, potrf, potrf_pk, potrf_pk3

comptime NEWLAPACK = "/System/Library/Frameworks/Accelerate.framework/Accelerate"


def raw(mut A: List[Float64]) -> Ptr:
    return A.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: bench-potrf neon|sme|accelerate-blas|PATH [--uplo=L|U]",
              "[--order=row|col] [--threads=T|auto] [--nb=NB] [n ...]")
        return
    var impl = String(args[1])
    var lower = True
    var row = False
    var nthreads = 1
    var nb = 256
    var pkmode = 0  # --pk: pack once, left-looking (small n, SME2);
    # --pk3: the same with lookahead on two threads
    var sizes = List[Int]()
    for i in range(2, len(args)):
        var a = String(args[i])
        if a == "--uplo=L" or a == "--uplo=U":
            lower = a == "--uplo=L"
        elif a == "--order=row" or a == "--order=col":
            row = a == "--order=row"
        elif a == "--pk":
            pkmode = 1
        elif a == "--pk3":
            pkmode = 3
        elif a == "--threads=auto":
            nthreads = 0  # potrf_select chooses nthreads and nb
        elif a.startswith("--threads="):
            nthreads = Int(a[byte=10:])
        elif a.startswith("--nb="):
            nb = Int(a[byte=5:])
        else:
            sizes.append(Int(a))
    if len(sizes) == 0:
        for n in range(50, 4001, 50):
            sizes.append(n)

    # performance cores (macOS: QOS_CLASS_USER_INTERACTIVE)
    _ = external_call["pthread_set_qos_class_self_np", Int32](
        UInt32(0x21), Int32(0)
    )

    var kind = 0 if impl == "neon" else (1 if impl == "sme" else 2)
    var acc = 0
    if impl == "accelerate-blas":
        kind = 1
        acc = 1
    var lib = OwnedDLHandle(impl if kind == 2 else String(NEWLAPACK))
    var symbol = String("dpotrf$NEWLAPACK") if impl == NEWLAPACK else String(
        "dpotrf_"
    )
    var dpotrf = lib.get_function[NoneType](symbol)

    print(String("# DPOTRF, ", "lower" if lower else "upper", ", ",
                 "row major" if row else "col major"))
    if kind < 2:
        var blas = String("Accelerate") if acc == 1 else String(
            "Mojo, SME2 micro kernel"
        ) if kind == 1 else String("Mojo, NEON micro kernel")
        print(String("# Mojo potrf, BLAS: ", blas, ", nb = ",
                     nb if nthreads > 0 else 0, ", threads = ",
                     String(nthreads) if nthreads > 0 else String("auto")))
    else:
        var slash = impl.rfind("/")
        print(String("# ", symbol, " of ", impl[byte=slash + 1 :]))
    print("# n seconds GFLOP/s check")

    seed(1)
    var ws = Workspace()
    var pkb = PotrfContext()
    for n in sizes:
        # A0 = M M^T / n + I (column major; symmetric, so also row major)
        var M = List[Float64](length=n * n, fill=0.0)
        for i in range(n * n):
            M[i] = random_float64(-1.0, 1.0)
        var A0 = List[Float64](length=n * n, fill=0.0)
        gemm[True](n, n, n, 1.0 / Float64(n), raw(M), 1, n, raw(M), n, 1,
                   0.0, raw(A0), 1, n, ws)
        for i in range(n):
            A0[i * n + i] += 1.0
        var A = A0.copy()
        var incRow = n if row else 1
        var incCol = 1 if row else n

        var ok = True
        var t = List[Float64]()
        for sample in range(4):
            var min_time = 0.3 if sample == 0 else 0.2
            var calls = 0
            var elapsed = 0.0
            while elapsed < min_time:
                for i in range(n * n):
                    A[i] = A0[i]
                var t0 = perf_counter_ns()
                var info: Int
                if pkmode > 0:
                    var r = incRow if lower else incCol
                    var c = incCol if lower else incRow
                    if pkmode == 1:
                        info = potrf_pk(n, raw(A), r, c, pkb)
                    else:
                        info = potrf_pk3(n, raw(A), r, c, pkb)
                elif kind == 0:
                    info = potrf[False](lower, n, raw(A), incRow, incCol, pkb,
                                        nb, nthreads)
                elif kind == 1 and acc == 0:
                    info = potrf[True](lower, n, raw(A), incRow, incCol, pkb,
                                       nb, nthreads)
                elif kind == 1:
                    info = potrf[True, 1](lower, n, raw(A), incRow, incCol,
                                          pkb, nb, nthreads)
                else:
                    # LAPACK is column major: row major lower = col major
                    # upper
                    var uplo = Int8(ord("L")) if lower != row else Int8(
                        ord("U")
                    )
                    var n_ = Int32(n)
                    var ld = Int32(n)
                    var info_ = Int32(0)
                    dpotrf(
                        Pointer(to=uplo), Pointer(to=n_), raw(A), Pointer(to=ld),
                        Pointer(to=info_),
                    )
                    info = Int(info_)
                elapsed += Float64(perf_counter_ns() - t0) * 1e-9
                if sample == 0 and calls == 0:
                    ok = info == 0 and check(n, lower, A, incRow, incCol, A0)
                calls += 1
            if sample > 0:
                t.append(elapsed / Float64(calls))
        sort(t)
        var gflops = Float64(n) ** 3 / 3.0 / t[1] * 1e-9
        print(n, t[1], gflops, "ok" if ok else "WRONG")
        _ = M^


# 400 random elements (i, j), i >= j, of L L^T against A0.
def check(n: Int, lower: Bool, A: List[Float64], incRow: Int, incCol: Int,
          A0: List[Float64]) -> Bool:
    comptime eps = 2.220446049250313e-16
    # element (i, j) of L at A[i*rl + j*cl]
    var rl = incRow if lower else incCol
    var cl = incCol if lower else incRow
    var amax = 0.0
    for x in A0:
        amax = max(amax, abs(x))
    for _ in range(400):
        var i = Int(random_ui64(0, UInt64(n - 1)))
        var j = Int(random_ui64(0, UInt64(n - 1)))
        if j > i:
            i, j = j, i
        var s = 0.0
        for l in range(j + 1):
            s += A[i * rl + l * cl] * A[j * rl + l * cl]
        if abs(s - A0[i * n + j]) > 8.0 * Float64(n) * eps * amax:
            return False
    return True
