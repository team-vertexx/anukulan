#!/bin/bash
# Where the time goes, by sampling. Written down because the decision about what
# to thread has to come from this rather than from intuition - and because the
# profile in the docs is old enough that it disagrees with this one.
set -e
cd "$(dirname "$0")/../.."
mkdir -p bench/results
out=bench/results/profiles.txt

sample_one() {
  local tag=$1 secs=$2; shift 2
  "$@" > /dev/null 2>&1 &
  local pid=$!
  sleep 1
  sample "$pid" "$secs" -mayDie -f "/tmp/sankhya_$tag.sample" > /dev/null 2>&1 || true
  wait "$pid" 2>/dev/null || true
  echo "===== $tag: $* ====="
  python3 - "/tmp/sankhya_$tag.sample" <<'PY'
import re, sys, collections
txt = open(sys.argv[1], errors="ignore").read()
counts = collections.Counter()
if "Sort by top of stack" in txt:
    for line in txt.split("Sort by top of stack", 1)[1].splitlines():
        m = re.match(r"\s*(\S.*?)\s{2,}(\d+)\s*$", line)
        if m:
            counts[m.group(1)] += int(m.group(2))
total = sum(counts.values())
print(f"{total} samples at top of stack")
for name, c in counts.most_common(18):
    print(f"{100*c/total:5.1f}%  {c:6d}  {name[:96]}")
PY
  echo
}

{
  echo "# load average at start: $(uptime | sed 's/.*averages: //')"
  echo
  sample_one milp_mas76 25 build/sankhya milp data/miplib/mas76.mps --time-limit=40
  sample_one simplex_degen3 25 build/sankhya simplex data/netlib/degen3.mps --presolve
  sample_one pdhg_datt256 25 build/sankhya solve data/lptestset/datt256_lp.mps --no-polish --max-iter=100000 --time-limit=40
  echo "# load average at end: $(uptime | sed 's/.*averages: //')"
} > "$out" 2>&1
echo "wrote $out"
