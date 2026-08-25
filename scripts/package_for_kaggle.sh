#!/usr/bin/env bash
# Package the source for upload to Kaggle as a private Dataset.
#
# Source only. The benchmark instances are several hundred megabytes and are
# downloaded on the Kaggle side by the fetch scripts, which is faster than
# uploading them and keeps the dataset small enough to re-upload freely.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
out="sankhya-source.zip"
rm -f "$out"

# git archive takes exactly what is tracked, so anything gitignored - build
# directories, downloaded instances - cannot leak in by accident.
git archive --format=zip --output="$out" HEAD
size=$(du -h "$out" | cut -f1)
files=$(unzip -l "$out" | tail -1 | awk '{print $2}')
echo "wrote $out  ($size, $files files)"
echo
echo "Next: kaggle.com -> Datasets -> New Dataset -> drag this file in,"
echo "title it sankhya-source, set it Private, Create."
echo "Then follow docs/KAGGLE.md from step 2."
