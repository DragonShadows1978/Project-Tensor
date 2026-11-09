#!/usr/bin/env bash
# Lead lane tool (contract C): seal a GPU run root at run end.
# usage: post_run_manifest.sh <run_root> <manifest_out>
# Writes "sha256  ./relative/path" for every regular file under <run_root> (symlinks are not followed and are
# listed separately), then re-checks the manifest against the bytes on disk. The lead scores the run only after
# this exits 0, and commits the manifest with the verdict. Prior art: GNU coreutils sha256sum -c; no new algorithm.
set -euo pipefail
root=${1:?run root}; out=${2:?manifest path}
[ -d "$root" ] || { echo "post_run_manifest: no run root: $root" >&2; exit 2; }
case "$out" in "$root"/*) echo "post_run_manifest: manifest must live outside the run root" >&2; exit 2;; esac
tmp="$out.partial"
( cd "$root" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum -- ) > "$tmp"
links=$(cd "$root" && find . -type l | LC_ALL=C sort)
if [ -n "$links" ]; then printf '%s\n' "$links" | sed 's/^/symlink  /' > "$out.symlinks"; fi
mv -f "$tmp" "$out"
( cd "$root" && sha256sum -c --quiet --strict -- "$out" )
n=$(wc -l < "$out")
echo "post_run_manifest: $n files sealed; manifest sha256 $(sha256sum "$out" | cut -d' ' -f1)"
