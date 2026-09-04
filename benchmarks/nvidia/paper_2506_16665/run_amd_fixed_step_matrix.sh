#!/usr/bin/env bash
set -euo pipefail

# Host-local reproducibility wrapper for the self-contained PR5 experiment.
# Override any path below in the environment when using another private stack.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-/tmp/meep-amd-fixed-step}
PYTHON=${PYTHON:-/home/alechammond/rocm-mpi/python/bin/python}
MPIEXEC=${MPIEXEC:-/home/alechammond/rocm-mpi/openmpi/bin/mpirun}
UCX_PREFIX=${UCX_PREFIX:-/home/alechammond/rocm-mpi/ucx}
ROCM_PREFIX=${ROCM_PREFIX:-/usr/local/fbcode/platform010/lib/rocm-latest}
MEEP_DEPS=${MEEP_DEPS:-/home/alechammond/meep-env}
MEEP_BUILD=${MEEP_BUILD:-/tmp/swarm-meep-rocm-mpi/build-pr5-python}
MEEP_PYTHONPATH=${MEEP_PYTHONPATH:-${MEEP_BUILD}/python}

mkdir -p "$OUT"
common=(
  --python "$PYTHON"
  --mpiexec "$MPIEXEC"
  --build-directory "$MEEP_BUILD"
  --library-path "$MEEP_PYTHONPATH"
  --runtime-library-path "$MEEP_BUILD/src/.libs"
  --runtime-library-path "$(dirname "$MPIEXEC")/../lib"
  --runtime-library-path "$UCX_PREFIX/lib"
  --runtime-library-path "$ROCM_PREFIX/lib"
  --runtime-library-path "$MEEP_DEPS/lib"
  --toolkit-compiler "$ROCM_PREFIX/bin/hipcc"
  --rocm-smi "$ROCM_PREFIX/bin/rocm-smi"
  --ucx-info "$UCX_PREFIX/bin/ucx_info"
)

run_case() {
  "$PYTHON" "$HERE/run_self_contained_benchmark.py" "$@" "${common[@]}"
}

run_case --output "$OUT/cpu-1thread.json" --backend cpu --route host --ranks 1 \
  --omp-threads 1 --map-by ppr:1:node --bind-to core
run_case --output "$OUT/cpu-8thread.json" --backend cpu --route host --ranks 1 \
  --omp-threads 8 --map-by ppr:1:package --bind-to package

CPU_REFERENCE=$(
  "$PYTHON" -c \
    'import json,sys; print(min(sys.argv[1:], key=lambda p: json.load(open(p))["timing"]["median_seconds"]))' \
    "$OUT/cpu-1thread.json" "$OUT/cpu-8thread.json"
)

run_case --output "$OUT/hip-np1.json" --backend nvidia --accelerator hip \
  --route direct --ranks 1 --visible-devices 1 --omp-threads 1 \
  --map-by ppr:1:package --rank-by fill --bind-to core --reference "$CPU_REFERENCE"

run_pair() {
  local ranks=$1 selectors=$2 mapping=$3
  run_case --output "$OUT/hip-staged-np${ranks}.json" --backend nvidia \
    --accelerator hip --route staged --ranks "$ranks" --visible-devices "$selectors" \
    --omp-threads 1 --map-by "$mapping" --rank-by fill --bind-to core \
    --reference "$CPU_REFERENCE"
  run_case --output "$OUT/hip-direct-np${ranks}.json" --backend nvidia \
    --accelerator hip --route direct --ranks "$ranks" --visible-devices "$selectors" \
    --omp-threads 1 --map-by "$mapping" --rank-by fill --bind-to core \
    --reference "$CPU_REFERENCE" \
    --peer-route-result "$OUT/hip-staged-np${ranks}.json"
}

run_pair 2 1,5 ppr:1:package
run_pair 4 1,3,5,7 ppr:2:package
run_pair 8 0,1,2,3,4,5,6,7 ppr:4:package
