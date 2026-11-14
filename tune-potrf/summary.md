Fastest configuration per n; a configuration with fewer threads is taken if it is within 3% of the fastest.

### sme

| n | algorithm | threads | nb | GFLOP/s |
|---:|---|---:|---:|---:|
| 50 | pk | 1 |  | 8 |
| 100 | pk3 | 2 |  | 23 |
| 150 | pk3 | 2 |  | 54 |
| 200 | pk3 | 2 |  | 81 |
| 250 | pk3 | 2 |  | 117 |
| 300 | pk3 | 2 |  | 142 |
| 400 | pk3 | 2 |  | 196 |
| 500 | pk3 | 2 |  | 241 |
| 650 | pk3 | 2 |  | 274 |
| 800 | pk3 | 2 |  | 326 |
| 1000 | pk3 | 2 |  | 331 |
| 1250 | pk3 | 2 |  | 329 |
| 1500 | pk3 | 2 |  | 369 |
| 1750 | pk | 1 |  | 348 |
| 2000 | pk | 1 |  | 356 |
| 2500 | pk | 1 |  | 365 |
| 3000 | tiles | 3 | 384 | 353 |
| 3500 | tiles | 3 | 256 | 352 |
| 4000 | tiles | 4 | 256 | 376 |

### accelerate-blas

| n | algorithm | threads | nb | GFLOP/s |
|---:|---|---:|---:|---:|
| 50 | tiles | 1 | 512 | 9 |
| 100 | tiles | 1 | 256 | 28 |
| 150 | tiles | 1 | 256 | 49 |
| 200 | tiles | 1 | 512 | 75 |
| 250 | tiles | 1 | 4096 | 91 |
| 300 | tiles | 1 | 512 | 112 |
| 400 | tiles | 1 | 4096 | 171 |
| 500 | tiles | 2 | 128 | 204 |
| 650 | tiles | 4 | 128 | 249 |
| 800 | tiles | 2 | 256 | 312 |
| 1000 | tiles | 2 | 256 | 355 |
| 1250 | tiles | 4 | 128 | 369 |
| 1500 | tiles | 3 | 256 | 383 |
| 1750 | tiles | 3 | 256 | 373 |
| 2000 | tiles | 4 | 128 | 391 |
| 2500 | tiles | 2 | 384 | 347 |
| 3000 | tiles | 4 | 512 | 362 |
| 3500 | tiles | 2 | 512 | 350 |
| 4000 | tiles | 4 | 512 | 368 |
