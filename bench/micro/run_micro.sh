#!/bin/bash
# Rebuilds the microbenchmark and records what the machine can actually do, into
# bench/results/ so the numbers outlive the shell that took them.
#
# Run this on as quiet a box as you can get, and read the load averages it
# prints: every number here is wall clock at a thread count, which is the one
# kind of measurement that a second busy process does not merely add noise to
# but inverts.
set -e
cd "$(dirname "$0")/../.."
mkdir -p bench/results
BIN=build/micro_spmv_scaling
c++ -std=c++20 -O2 -Isrc -o "$BIN" bench/micro/spmv_scaling.cpp build/libsankhya.a

out=bench/results/spmv_scaling.txt
{
  echo "# machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.perflevel0.logicalcpu) performance + $(sysctl -n hw.perflevel1.logicalcpu) efficiency cores"
  echo "# load average at start: $(uptime | sed 's/.*averages: //')"
  echo
  for f in data/lptestset/datt256_lp.mps data/lptestset/supportcase10.mps \
           data/netlib/maros-r7.mps data/netlib/25fv47.mps; do
    echo "===== $(basename "$f" .mps) ====="
    "$BIN" "$f" 10
    echo
  done
  echo "# load average at end: $(uptime | sed 's/.*averages: //')"
} > "$out" 2>&1
echo "wrote $out"
