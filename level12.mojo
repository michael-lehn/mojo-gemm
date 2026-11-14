# BLAS level 1 and level 2 operations in Mojo, in place, for general
# strides (element i of x at x[i*incX], element (i, j) of A at
# A[i*incRowA + j*incColA]).  Unit strides take SIMD (single instruction,
# multiple data) paths, all other strides plain loops.  Used by the
# unblocked kernels of potrf.

from gemm import Ptr

comptime V = 8  # SIMD width (doubles)
comptime f64 = DType.float64


# x^T y
def dot(n: Int, x: Ptr, incX: Int, y: Ptr, incY: Int) -> Float64:
    if incX == 1 and incY == 1:
        var s0 = SIMD[f64, V](0.0)
        var s1 = SIMD[f64, V](0.0)
        var i = 0
        while i + 2 * V <= n:
            s0 += x.unsafe_load[width=V](i) * y.unsafe_load[width=V](i)
            s1 += x.unsafe_load[width=V](i + V) * y.unsafe_load[width=V](
                i + V
            )
            i += 2 * V
        var s = (s0 + s1).reduce_add()
        while i < n:
            s += x[unsafe_offset=i] * y[unsafe_offset=i]
            i += 1
        return s
    var s = 0.0
    for i in range(n):
        s += x[unsafe_offset=i * incX] * y[unsafe_offset=i * incY]
    return s


# x <- alpha x
def scal(n: Int, alpha: Float64, x: Ptr, incX: Int):
    if incX == 1:
        var i = 0
        while i + V <= n:
            x.unsafe_store(i, alpha * x.unsafe_load[width=V](i))
            i += V
        while i < n:
            x[unsafe_offset=i] *= alpha
            i += 1
        return
    for i in range(n):
        x[unsafe_offset=i * incX] *= alpha


# y <- y + alpha A x, A is m x n
def gemv(
    m: Int, n: Int, alpha: Float64, A: Ptr, incRowA: Int, incColA: Int,
    x: Ptr, incX: Int, y: Ptr, incY: Int,
):
    if m <= 0 or n <= 0:
        return
    if incRowA == 1 and incY == 1:
        # column by column (axpy), four columns per pass over y
        var j = 0
        while j + 4 <= n:
            var a0 = A.unsafe_offset(j * incColA)
            var a1 = A.unsafe_offset((j + 1) * incColA)
            var a2 = A.unsafe_offset((j + 2) * incColA)
            var a3 = A.unsafe_offset((j + 3) * incColA)
            var b0 = alpha * x[unsafe_offset=j * incX]
            var b1 = alpha * x[unsafe_offset=(j + 1) * incX]
            var b2 = alpha * x[unsafe_offset=(j + 2) * incX]
            var b3 = alpha * x[unsafe_offset=(j + 3) * incX]
            var i = 0
            while i + V <= m:
                var v = y.unsafe_load[width=V](i)
                v += b0 * a0.unsafe_load[width=V](i)
                v += b1 * a1.unsafe_load[width=V](i)
                v += b2 * a2.unsafe_load[width=V](i)
                v += b3 * a3.unsafe_load[width=V](i)
                y.unsafe_store(i, v)
                i += V
            while i < m:
                y[unsafe_offset=i] += (
                    b0 * a0[unsafe_offset=i] + b1 * a1[unsafe_offset=i]
                    + b2 * a2[unsafe_offset=i] + b3 * a3[unsafe_offset=i]
                )
                i += 1
            j += 4
        while j < n:
            var a0 = A.unsafe_offset(j * incColA)
            var b0 = alpha * x[unsafe_offset=j * incX]
            var i = 0
            while i + V <= m:
                y.unsafe_store(
                    i, y.unsafe_load[width=V](i) + b0 * a0.unsafe_load[width=V](i)
                )
                i += V
            while i < m:
                y[unsafe_offset=i] += b0 * a0[unsafe_offset=i]
                i += 1
            j += 1
        return
    # row by row (dot products; SIMD if the rows of A and x are contiguous)
    for i in range(m):
        y[unsafe_offset=i * incY] += alpha * dot(
            n, A.unsafe_offset(i * incRowA), incColA, x, incX
        )

