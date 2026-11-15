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
#      NUMAX32=<numax float32 benchmark binary> \
#          sh run-potrf-benchmarks.sh --float [n ...]   (SPOTRF, in
#                                                       results-potrf-float/)
#
#  NUMAX: bench_numax_cholesky.mojo of this repository built in a numax
#  checkout (see there).  Without NUMAX numax is skipped.
#
#  PAUSE=<seconds>: pause before each run, so that the MacBook Air (no
#  fan) can cool down; default 0.
#
#  OPENBLAS=<path of libopenblas.dylib>: also OpenBLAS's dpotrf (spotrf),
#  with its default number of threads (openblas.txt) and with one thread
#  (openblas-1.txt).  Without OPENBLAS OpenBLAS is skipped.
#
set -e
cd "$(dirname "$0")"
pixi run build-potrf
R=results-potrf
F=""
if [ "$1" = "--float" ]; then
    R=results-potrf-float
    F="--float"
    NUMAX="$NUMAX32"  # float32: needs Apple's Metal toolchain to build
    shift
fi
ACCELERATE=/System/Library/Frameworks/Accelerate.framework/Accelerate
pause() { sleep "${PAUSE:-0}"; }
export VECLIB_MAXIMUM_THREADS=1   # Accelerate's BLAS inside our threads
for order in col row; do
    D=$R/$order
    mkdir -p "$D"
    pause
    ./bench-potrf sme $F --threads=auto --order=$order "$@" > "$D/mojo.txt"
    pause
    ./bench-potrf accelerate-blas $F --threads=auto --order=$order "$@" \
        > "$D/mojo-accelerate-blas.txt"
    pause
    ./bench-potrf "$ACCELERATE" $F --order=$order "$@" > "$D/accelerate.txt"
    if [ -n "$OPENBLAS" ]; then
        pause
        ./bench-potrf "$OPENBLAS" $F --order=$order "$@" > "$D/openblas.txt"
        pause
        OPENBLAS_NUM_THREADS=1 ./bench-potrf "$OPENBLAS" $F --order=$order \
            "$@" > "$D/openblas-1.txt"
    fi
done
if [ -n "$NUMAX" ]; then
    "$NUMAX" "$@" > $R/numax.txt
fi
