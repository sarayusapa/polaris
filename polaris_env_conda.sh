# Source before running PolaRiS from the conda env:  conda activate polaris && source polaris_env_conda.sh
# Conda supplies python headers + the CUDA 12.8 toolkit natively, so no CPATH/pip-wheel hacks
# are needed (unlike the uv venv, see polaris_env.sh).
POLARIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[polaris_env_conda] ERROR: no conda env active. Run: conda activate polaris" >&2
  return 1 2>/dev/null || exit 1
fi

# Accept the Omniverse Kit / Isaac Sim EULA non-interactively (first-run gate)
export OMNI_KIT_ACCEPT_EULA=YES

# CUDA toolkit lives in the conda env (nvcc + headers, consistent versions)
export CUDA_HOME="$CONDA_PREFIX"

# Shims (still relevant with torch in-process alongside Isaac Sim's LLVM):
#  - triton/: blocks libtriton.so, whose LLVM static-init collides with Isaac Sim's LLVM
#  - sitecustomize.py: backfills sys.get_int_max_str_digits for torch's dynamo polyfill
export PYTHONPATH="$POLARIS_ROOT/.polaris_shims${PYTHONPATH:+:$PYTHONPATH}"

# Build the JIT CUDA extensions with the SYSTEM compiler, not conda's gcc 14.
# Conda's gcc links against libstdc++ 15 (needs GLIBCXX_3.4.32), but Isaac Sim's process
# loads Ubuntu 22.04's system libstdc++ (max GLIBCXX_3.4.30) -> ImportError at runtime.
# System g++ 11.4 produces GLIBCXX_3.4.21, which loads fine inside Isaac Sim.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Bundled static ffmpeg (mediapy video writing)
export PATH="$CUDA_HOME/bin:$POLARIS_ROOT/.polaris_bin:$PATH"

echo "[polaris_env_conda] env=$CONDA_DEFAULT_ENV CUDA_HOME=$CUDA_HOME"
echo "[polaris_env_conda] nvcc: $(command -v nvcc)  ffmpeg: $(command -v ffmpeg)"
