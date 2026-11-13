# Benchmark of DGEMM: C <- beta C + alpha A B, m = n = k.
#
#   ./bench IMPL [--alpha=X] [--beta=X] [--A=row|col] [--B=row|col]
#                [--C=row|col] [n ...]
#
#   IMPL   neon        Mojo, NEON micro kernel
#          sme         Mojo, SME2 micro kernel
#          PATH        cblas_dgemm of the shared library PATH (e.g. Apple
#                      Accelerate or ulmBLAS), for comparison
#   defaults: alpha = 1, beta = 0, A, B and C row major,
#             n = 50, 100, ..., 4000
#
# The protocol of ulmBLAS's bench/gemm/benchx.c: per n one call checked
# against sums for 400 random elements of C (error bound 4 (k + 2) eps
# (|alpha| sum |A(i,l) B(l,j)| + |beta C0(i,j)|)), which is also the
# warm-up, extended to at least 0.3 s; then 3 samples of at least 0.2 s
# each, the median is reported.  One thread, on a performance core (QoS
# class user-interactive on macOS).

from std.ffi import OwnedDLHandle, external_call
from std.os import getenv
from std.random import random_float64, random_ui64, seed
from std.sys import argv
from std.time import perf_counter_ns

from gemm import NEON, SME2, Ptr, Workspace, gemm

comptime ACCELERATE = (
    "/System/Library/Frameworks/Accelerate.framework/Accelerate"
)
# CBLAS constants
comptime CblasRowMajor = Int32(101)
comptime CblasColMajor = Int32(102)
comptime CblasNoTrans = Int32(111)
comptime CblasTrans = Int32(112)


def raw(mut A: List[Float64]) -> Ptr:
    return A.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


# "C = A*B", "C = C + A*B", "C = 0.5*C + 2.0*A*B", ...
def title(alpha: Float64, beta: Float64, row_A: Bool, row_B: Bool,
          row_C: Bool) -> String:
    var ab = String("A*B") if alpha == 1.0 else String(alpha, "*A*B")
    var s = String("C = ")
    if beta == 1.0:
        s += "C + "
    elif beta != 0.0:
        s += String(beta, "*C + ")
    s += ab

    def order(row: Bool) -> String:
        return "row major" if row else "col major"

    return String(
        "DGEMM ", s, ", A ", order(row_A), ", B ", order(row_B), ", C ",
        order(row_C),
    )


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: bench neon|sme|PATH [--alpha=X] [--beta=X]",
              "[--A=row|col] [--B=row|col] [--C=row|col] [n ...]")
        return
    var impl = String(args[1])
    var alpha = 1.0
    var beta = 0.0
    var row_A = True
    var row_B = True
    var row_C = True
    var sizes = List[Int]()
    for i in range(2, len(args)):
        var a = String(args[i])
        if a.startswith("--alpha="):
            alpha = atof(a[byte=8:])
        elif a.startswith("--beta="):
            beta = atof(a[byte=7:])
        elif a.startswith("--A=") or a.startswith("--B=") or a.startswith(
            "--C="
        ):
            var v = String(a[byte=4:])
            if v != "row" and v != "col":
                raise Error("storage order must be row or col: " + a)
            if a.startswith("--A="):
                row_A = v == "row"
            elif a.startswith("--B="):
                row_B = v == "row"
            else:
                row_C = v == "row"
        else:
            sizes.append(Int(a))
    if len(sizes) == 0:
        for n in range(50, 4001, 50):
            sizes.append(n)

    # one thread on a performance core (macOS: QOS_CLASS_USER_INTERACTIVE)
    _ = external_call["pthread_set_qos_class_self_np", Int32](
        UInt32(0x21), Int32(0)
    )

    # kind: 0 Mojo NEON, 1 Mojo SME2, 2 cblas_dgemm of a shared library
    var kind = 0 if impl == "neon" else (1 if impl == "sme" else 2)
    var lib = OwnedDLHandle(impl if kind == 2 else String(ACCELERATE))
    var cblas_dgemm = lib.get_function[NoneType]("cblas_dgemm")

    print("#", title(alpha, beta, row_A, row_B, row_C))
    if kind == 0:
        print(String("# Mojo, NEON micro kernel, MR x NR = ", NEON.MR, " x ",
                     NEON.NR, ", MC KC NC = ", NEON.MC, " ", NEON.KC, " ",
                     NEON.NC))
    elif kind == 1:
        print(String("# Mojo, SME2 micro kernel, MR x NR = ", SME2.MR, " x ",
                     SME2.NR, ", MC KC NC = ", SME2.MC, " ", SME2.KC, " ",
                     SME2.NC))
    else:
        var backend = getenv("ULMBLAS_BACKEND")
        var slash = impl.rfind("/")  # only the file name of the library
        print(String("# cblas_dgemm of ", impl[byte=slash + 1 :],
                     "" if backend == "" else String(", ULMBLAS_BACKEND=",
                                                      backend)))
    print("# n seconds GFLOP/s check")

    seed(1)
    var ws = Workspace()
    for n in sizes:
        var A = List[Float64](length=n * n, fill=0.0)
        var B = List[Float64](length=n * n, fill=0.0)
        var C = List[Float64](length=n * n, fill=0.0)
        for i in range(n * n):
            A[i] = random_float64(-1.0, 1.0)
            B[i] = random_float64(-1.0, 1.0)
            C[i] = random_float64(-1.0, 1.0)
        var C0 = C.copy()
        # strides of the square matrices
        var incRowA = n if row_A else 1
        var incColA = 1 if row_A else n
        var incRowB = n if row_B else 1
        var incColB = 1 if row_B else n
        var incRowC = n if row_C else 1
        var incColC = 1 if row_C else n
        # the same product with CBLAS: the layout of C, A and B transposed
        # if their storage order differs from C's (all leading dimensions n)
        var layout = CblasRowMajor if row_C else CblasColMajor
        var transA = CblasNoTrans if row_A == row_C else CblasTrans
        var transB = CblasNoTrans if row_B == row_C else CblasTrans

        var ok = True
        var t = List[Float64]()
        # sample 0: the checked call and the warm-up, then 3 samples
        for sample in range(4):
            var min_time = 0.3 if sample == 0 else 0.2
            var calls = 0
            var t0 = perf_counter_ns()
            var elapsed = 0.0
            while elapsed < min_time:
                if kind == 0:
                    gemm[False](
                        n, n, n, alpha, raw(A), incRowA, incColA, raw(B),
                        incRowB, incColB, beta, raw(C), incRowC, incColC, ws,
                    )
                elif kind == 1:
                    gemm[True](
                        n, n, n, alpha, raw(A), incRowA, incColA, raw(B),
                        incRowB, incColB, beta, raw(C), incRowC, incColC, ws,
                    )
                else:
                    cblas_dgemm(
                        layout, transA, transB, Int32(n), Int32(n), Int32(n),
                        alpha, raw(A), Int32(n), raw(B), Int32(n), beta,
                        raw(C), Int32(n),
                    )
                if sample == 0 and calls == 0:
                    ok = check(n, alpha, beta, A, incRowA, incColA, B,
                               incRowB, incColB, C0, C, incRowC, incColC)
                    t0 = perf_counter_ns()  # the check is not timed
                calls += 1
                elapsed = Float64(perf_counter_ns() - t0) * 1e-9
            if sample > 0:
                t.append(elapsed / Float64(calls))
        sort(t)
        var gflops = 2.0 * Float64(n) ** 3 / t[1] * 1e-9
        print(n, t[1], gflops, "ok" if ok else "WRONG")


# 400 random elements of C = beta C0 + alpha A B against sums with the
# error bound 4 (k + 2) eps (|alpha| sum |A(i,l) B(l,j)| + |beta C0(i,j)|).
def check(
    n: Int, alpha: Float64, beta: Float64,
    A: List[Float64], incRowA: Int, incColA: Int,
    B: List[Float64], incRowB: Int, incColB: Int,
    C0: List[Float64], C: List[Float64], incRowC: Int, incColC: Int,
) -> Bool:
    comptime eps = 2.220446049250313e-16
    for _ in range(400):
        var i = Int(random_ui64(0, UInt64(n - 1)))
        var j = Int(random_ui64(0, UInt64(n - 1)))
        var s = 0.0
        var bound = 0.0
        for l in range(n):
            var p = A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB]
            s += p
            bound += abs(p)
        var cij = alpha * s
        bound = abs(alpha) * bound
        if beta != 0.0:
            var c0 = C0[i * incRowC + j * incColC]
            cij += beta * c0
            bound += abs(beta * c0)
        if abs(C[i * incRowC + j * incColC] - cij) > 4.0 * Float64(
            n + 2
        ) * eps * bound:
            return False
    return True
