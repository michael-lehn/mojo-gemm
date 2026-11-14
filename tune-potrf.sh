#!/bin/sh
#
#  Setup step for a new machine (as in ATLAS): benchmarks potrf for a grid
#  of n, both BLAS backends, the tiled algorithm with all thread counts and
#  tile sizes nb and (Mojo BLAS) the pack-once algorithms, then
#  generates potrf_select.mojo, the function potrf uses for nthreads = 0
#  (see gen_potrf_select.py).  Takes about half an hour on an Apple M5.
#
#      sh tune-potrf.sh
#
#  Results: tune-potrf/<backend>/t<threads>-nb<nb>.txt and
#  tune-potrf/summary.md.  Afterwards rebuild (pixi run build-potrf).
#
set -e
cd "$(dirname "$0")"
pixi run build-potrf
N="50 100 150 200 250 300 400 500 650 800 1000 1250 1500 1750 2000 2500 3000 3500 4000"
PCORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)
ALL=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)
THREADS="2 3 $PCORES $ALL"
export VECLIB_MAXIMUM_THREADS=1   # Accelerate's kernels inside our tasks
for backend in sme accelerate-blas; do
    D=tune-potrf/$backend
    rm -rf "$D"
    mkdir -p "$D"
    if [ $backend = sme ]; then
        echo "$backend: pack once, pack once with lookahead"
        ./bench-potrf sme --pk $N > "$D/pk.txt"
        ./bench-potrf sme --pk3 $N > "$D/pk3.txt"
    fi
    for nb in 256 512 4096; do
        echo "$backend: 1 thread, nb = $nb"
        ./bench-potrf $backend --threads=1 --nb=$nb $N > "$D/t1-nb$nb.txt"
    done
    for t in $(echo $THREADS | tr ' ' '\n' | sort -nu); do
        for nb in 128 256 384 512; do
            echo "$backend: $t threads, nb = $nb"
            ./bench-potrf $backend --threads=$t --nb=$nb $N > "$D/t$t-nb$nb.txt"
        done
    done
done
python3 gen_potrf_select.py
