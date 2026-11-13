#!/bin/sh
#
#  Runs all benchmarks: the four storage orders of A and B (C row major,
#  alpha = 1, beta = 0), each for the Mojo NEON and SME2 micro kernels,
#  ulmBLAS (NEON and SME2 backends) and Apple Accelerate.  Results in
#  results/A-<order>_B-<order>/<implementation>.txt.
#
#      ULMBLAS=<prefix of 'make install-shared' of ulmBLAS> sh run-benchmarks.sh
#
#  ulmBLAS: git clone https://github.com/michael-lehn/ulmBLAS
#           cd ulmBLAS && make install-shared PREFIX=<prefix>
#  Without ULMBLAS the ulmBLAS runs are skipped.
#
set -e
cd "$(dirname "$0")"
pixi run mojo build -O3 -I . bench.mojo -o bench

ACCELERATE=/System/Library/Frameworks/Accelerate.framework/Accelerate

for A in row col; do
    for B in row col; do
        D=results/A-${A}_B-${B}
        mkdir -p "$D"
        opt="--A=$A --B=$B"
        echo "A $A major, B $B major"
        ./bench neon $opt "$@" > "$D/mojo-neon.txt"
        ./bench sme $opt "$@" > "$D/mojo-sme2.txt"
        if [ -n "$ULMBLAS" ]; then
            OMP_NUM_THREADS=1 ULMBLAS_BACKEND=neon ./bench "$ULMBLAS/lib/libulmblas.dylib" \
                $opt "$@" > "$D/ulmblas-neon.txt"
            OMP_NUM_THREADS=1 ULMBLAS_BACKEND=sme ./bench "$ULMBLAS/lib/libulmblas.dylib" \
                $opt "$@" > "$D/ulmblas-sme2.txt"
        fi
        VECLIB_MAXIMUM_THREADS=1 ./bench "$ACCELERATE" $opt "$@" \
            > "$D/accelerate.txt"
    done
done
