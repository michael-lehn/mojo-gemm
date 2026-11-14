#!/bin/sh
#
#  DPOTRF benchmarks (lower), column major and row major:
#    mojo.txt                   Mojo potrf, Mojo BLAS (SME2), configuration
#                               from potrf_select (tuning step)
#    mojo-accelerate-blas.txt   Mojo potrf, BLAS of Accelerate, from
#                               potrf_select
#    accelerate.txt             Accelerate's dpotrf (one thread)
#  and numax (row major only).  Results in results-potrf/<order>/.
#
#      NUMAX=<numax benchmark binary> sh run-potrf-benchmarks.sh [n ...]
#      sh run-potrf-benchmarks.sh --float [n ...]   (SPOTRF, in
#                                                   results-potrf-float/)
#
#  NUMAX: bench_numax_cholesky.mojo of this repository built in a numax
#  checkout (see there).  Without NUMAX numax is skipped.
#
set -e
cd "$(dirname "$0")"
pixi run build-potrf
R=results-potrf
F=""
if [ "$1" = "--float" ]; then
    R=results-potrf-float
    F="--float"
    NUMAX=""  # numax: float32 needs Apple's Metal toolchain to build
    shift
fi
ACCELERATE=/System/Library/Frameworks/Accelerate.framework/Accelerate
export VECLIB_MAXIMUM_THREADS=1   # Accelerate's BLAS inside our threads
for order in col row; do
    D=$R/$order
    mkdir -p "$D"
    ./bench-potrf sme $F --threads=auto --order=$order "$@" > "$D/mojo.txt"
    ./bench-potrf accelerate-blas $F --threads=auto --order=$order "$@" \
        > "$D/mojo-accelerate-blas.txt"
    ./bench-potrf "$ACCELERATE" $F --order=$order "$@" > "$D/accelerate.txt"
done
if [ -n "$NUMAX" ]; then
    "$NUMAX" "$@" > results-potrf/numax.txt
fi
