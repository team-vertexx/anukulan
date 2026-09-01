#!/bin/bash
# The answers-did-not-change deliverable, end to end.
#
# The two harnesses that check against published optima both reach the
# first-order method - verify_simplex.py through the crossover seed, and
# miplib_survey.py through the root crossover - so running them with --threads
# on and off is a real test of the parallel path rather than a formality.
# threaded_cpu_backend is bit-identical to the serial one by construction, so
# the pass is that the two outputs are the same file, not merely two files that
# both look healthy.
set -e
cd "$(dirname "$0")/.."
mkdir -p bench/results
T=${1:-6}

echo "== simplex, 88 Netlib instances against published optima =="
python3 bench/verify_simplex.py 45 --presolve --crossover \
  > bench/results/verify_simplex_serial.txt 2>&1 || true
python3 bench/verify_simplex.py 45 --presolve --crossover --threads="$T" \
  > bench/results/verify_simplex_threads$T.txt 2>&1 || true

echo "== milp survey =="
python3 bench/miplib_survey.py 15 \
  > bench/results/miplib_serial.txt 2>&1 || true
python3 bench/miplib_survey.py 15 --threads="$T" \
  > bench/results/miplib_threads$T.txt 2>&1 || true

echo
for pair in "verify_simplex_serial.txt verify_simplex_threads$T.txt" \
            "miplib_serial.txt miplib_threads$T.txt"; do
  set -- $pair
  # The survey prints its own wall clock, which is the one thing that must
  # differ. Comparing everything else is the point.
  if diff <(grep -v '[0-9]\.[0-9]s' "bench/results/$1") \
          <(grep -v '[0-9]\.[0-9]s' "bench/results/$2") > /dev/null; then
    echo "IDENTICAL   $1 vs $2"
  else
    echo "DIFFERS     $1 vs $2"
    diff <(grep -v '[0-9]\.[0-9]s' "bench/results/$1") \
         <(grep -v '[0-9]\.[0-9]s' "bench/results/$2") | head -30
  fi
done
