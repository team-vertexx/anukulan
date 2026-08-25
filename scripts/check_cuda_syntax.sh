#!/usr/bin/env bash
# Type-check the CUDA backend on a machine that has no nvcc.
#
#   bash scripts/check_cuda_syntax.sh
#
# The CUDA backend is written on a laptop and compiled on a GPU box, which means
# a typo costs a whole round trip - and one round trip was already spent that
# way. This catches everything a compiler catches except the parts that are
# genuinely CUDA: it stubs the runtime, strips the <<<grid, block>>> launch
# configurations, which are not C++, and hands the rest to clang.
#
# What it does NOT check: that the kernels compute the right thing, that shared
# memory fits, that the launch shapes are sane, or anything about a device. It
# checks that the file compiles. That is the failure worth catching from here.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cp "$root/scripts/cudastub/cuda_runtime.h" "$work/"
python3 - "$root/src/sankhya/cuda_backend.cu" "$work/backend_check.cpp" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
out = re.sub(r'<<<.*?>>>', '', src, flags=re.S)
out += "\nuint3_t blockIdx{0,0,0}, threadIdx{0,0,0}, blockDim{256,1,1}, gridDim{1,1,1};\n"
open(sys.argv[2], 'w').write(out)
print(f"stripped {len(re.findall(r'<<<.*?>>>', src, flags=re.S))} kernel launches")
PY

compiler="${CXX:-clang++}"
command -v "$compiler" > /dev/null || compiler=g++
if "$compiler" -std=c++20 -fsyntax-only -I"$work" -I"$root/src" \
     "$work/backend_check.cpp" 2> "$work/errors.txt"; then
  echo "cuda_backend.cu type-checks clean"
  exit 0
fi
# 'grid' reads as unused once its launch configuration is stripped, which is an
# artefact of this check rather than a problem with the file.
grep -v "unused variable 'grid'" "$work/errors.txt" | head -40
echo
echo "cuda_backend.cu does not compile. Fix it before uploading to Kaggle."
exit 1
