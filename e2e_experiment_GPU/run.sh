#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build="${FELID_BUILD_DIR:-$root/build}"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release "$@" >&2
cmake --build "$build" --target felid_e2e -j "${FELID_BUILD_JOBS:-4}" >&2
exec "$build/felid_e2e"
