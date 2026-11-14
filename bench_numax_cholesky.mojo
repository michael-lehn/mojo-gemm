# Benchmark of numax's Cholesky factorization (numax.linalg.cholesky, CPU,
# run-time shape, row major) in the format of bench-potrf:
#
#   n seconds GFLOP/s check
#
# Build in a numax checkout (https://github.com/ehsanmok/numax):
#
#   cp bench_numax_cholesky.mojo <numax>/bench/
#   cd <numax> && pixi run mojo build -O3 -I . \
#       bench/bench_numax_cholesky.mojo -o bench-numax-cholesky
#   ./bench-numax-cholesky [n ...]        (default n = 50, 100, ..., 4000)
#
# DTYPE: float64 (change the comptime below for float32).  A(i, j) = 1 / (1
# + |i - j|), A(i, i) = n (positive definite).  Per n one checked call
# (400 random elements of L L^T against A), then 3 samples of at least
# 0.2 s each, the median is reported.  The time includes what numax's API
# does per call (the result is a new tensor).  GFLOP/s from n^3 / 3.

from std.benchmark import keep
from std.random import random_ui64, seed
from std.sys import argv
from std.time import perf_counter_ns

from layout.tile_layout import row_major

from numax.core.tensor import Dynamic, _dyn_shape
from numax.linalg import cholesky

comptime dtype = DType.float64


def entry(i: Int, j: Int, n: Int) -> Float64:
    return Float64(n) if i == j else 1.0 / (1.0 + Float64(abs(i - j)))


def main() raises:
    var args = argv()
    var sizes = List[Int]()
    for i in range(1, len(args)):
        sizes.append(Int(String(args[i])))
    if len(sizes) == 0:
        for n in range(50, 4001, 50):
            sizes.append(n)

    print(String("# numax cholesky, ", dtype, ", row major, run-time shape"))
    print("# n seconds GFLOP/s check")
    seed(1)
    for n in sizes:
        var v = List[Scalar[dtype]](capacity=n * n)
        for i in range(n):
            for j in range(n):
                v.append(Scalar[dtype](entry(i, j, n)))
        var a = Dynamic[dtype, 2](row_major(_dyn_shape[2](n, n)), v^)

        # check: 400 random elements of L L^T
        var l = cholesky(a).to_host()
        comptime eps = 2.220446049250313e-16 if dtype == DType.float64 else (
            1.1920928955078125e-07
        )
        var ok = True
        for _ in range(400):
            var i = Int(random_ui64(0, UInt64(n - 1)))
            var j = Int(random_ui64(0, UInt64(n - 1)))
            var s = 0.0
            for k in range(min(i, j) + 1):
                s += Float64(l[i * n + k]) * Float64(l[j * n + k])
            if abs(s - entry(i, j, n)) > 8.0 * Float64(n) * eps * Float64(n):
                ok = False

        var t = List[Float64]()
        for _ in range(3):
            var calls = 0
            var elapsed = 0.0
            while elapsed < 0.2:
                var t0 = perf_counter_ns()
                var r = cholesky(a)
                keep(r.tile().ptr)
                elapsed += Float64(perf_counter_ns() - t0) * 1e-9
                calls += 1
            t.append(elapsed / Float64(calls))
        sort(t)
        print(n, t[1], Float64(n) ** 3 / 3.0 / t[1] * 1e-9,
              "ok" if ok else "WRONG")
