#!/bin/bash
# Upload the weights of weights/MANIFEST_final.tsv to a Hugging Face model repository (run by the repository owner).
#
#   HF_REPO=<user>/<repo> bash tools/hf_stage_upload.sh upload SOURCES.tsv [pipe|server]
#   HF_REPO=<user>/<repo> bash tools/hf_stage_upload.sh card                      (model card + manifest)
#   HF_REPO=<user>/<repo> bash tools/hf_stage_upload.sh verify                    (sha256 of every LFS file on the Hub)
#
# SOURCES.tsv (kept outside the repository; it names the machines that hold the files) has the columns
#   file  host  path  size_bytes  sha256
# file = path inside the Hugging Face repository (= column `file` of the manifest); host = an ssh alias or `local`;
# path = path on that host (relative to $HOME, or relative to $LOCAL_ROOT for `local`; LOCAL_ROOT defaults to the
#   current directory, so set it when you run the script from somewhere else).
# Modes:
#   pipe   (default) each file is streamed over ssh into a temporary file on this machine (one file at a time, at most
#          about 1.2 GB of free disk needed), its sha256 is checked against the manifest, it is uploaded, the copy is deleted.
#          The Hugging Face token stays on this machine.
#   server the remote host runs `sha256sum` and `hf upload` itself (needs huggingface_hub and a login on that
#          host; do not leave a token on a shared machine).
# The Hub repository is created private on the first upload (--private); switch it to public by hand after the deadline.
# Every uploaded file is recorded in $STATE (default .hf_upload_done.txt), so a rerun continues where it stopped.
# Needs: the `hf` CLI of huggingface_hub (pip install "huggingface_hub<2"; `hf auth login` once, or HF_TOKEN), ssh, shasum or sha256sum.
set -euo pipefail
cmd=${1:?usage: $0 upload SOURCES.tsv [pipe|server] | card | verify}
: "${HF_REPO:?set HF_REPO=<user>/<repo>}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST=${MANIFEST:-$HERE/weights/MANIFEST_final.tsv}
STATE=${STATE:-$HERE/.hf_upload_done.txt}
LOCAL_ROOT=${LOCAL_ROOT:-$PWD}
HF_CLI=${HF_CLI:-hf}
TMPD=${TMPD:-${TMPDIR:-/tmp}/hf_stage.$$}
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
want() { awk -F'\t' -v f="$1" 'NR > 1 && $1 == f {print $6}' "$MANIFEST"; }

case $cmd in
upload)
  SRC=${2:?SOURCES.tsv}; MODE=${3:-pipe}
  mkdir -p "$TMPD"; touch "$STATE"; trap 'rm -rf "$TMPD"' EXIT
  n=0; skip=0
  while IFS=$'\t' read -r file host path size digest; do
    [ "$file" = file ] && continue
    grep -qxF "$file" "$STATE" && { skip=$((skip + 1)); continue; }
    w=$(want "$file"); [ -n "$w" ] || { echo "not in the manifest: $file" >&2; exit 1; }
    [ "$w" = "$digest" ] || { echo "sha256 of $file differs between SOURCES and the manifest" >&2; exit 1; }
    echo "== $file ($host, $size bytes)"
    if [ "$host" = local ]; then
      f="$LOCAL_ROOT/$path"
      [ -f "$f" ] || { echo "local file not found: $f (set LOCAL_ROOT to the directory the local paths in SOURCES are relative to)" >&2; exit 1; }
      got=$(sha "$f")
      [ "$got" = "$w" ] || { echo "sha256 mismatch $file: $got" >&2; exit 1; }
      "$HF_CLI" upload "$HF_REPO" "$f" "$file" --repo-type model --private --commit-message "add $file" </dev/null
    elif [ "$MODE" = server ]; then
      got=$(ssh "$host" "sha256sum \"\$HOME/$path\"" </dev/null | cut -d' ' -f1)
      [ "$got" = "$w" ] || { echo "sha256 mismatch $file on $host: $got" >&2; exit 1; }
      ssh "$host" "$HF_CLI upload '$HF_REPO' \"\$HOME/$path\" '$file' --repo-type model --private --commit-message 'add $file'" </dev/null
    else
      t="$TMPD/$(basename "$file")"
      ssh "$host" "cat \"\$HOME/$path\"" </dev/null > "$t"
      got=$(sha "$t")
      [ "$got" = "$w" ] || { echo "sha256 mismatch $file (copy from $host): $got" >&2; rm -f "$t"; exit 1; }
      "$HF_CLI" upload "$HF_REPO" "$t" "$file" --repo-type model --private --commit-message "add $file" </dev/null
      rm -f "$t"
    fi
    echo "$file" >> "$STATE"; n=$((n + 1))
  done < "$SRC"
  echo "uploaded $n, already done $skip (state: $STATE)"
  ;;
card)
  "$HF_CLI" upload "$HF_REPO" "$HERE/weights/HF_MODEL_CARD.md" README.md --repo-type model --commit-message "model card"
  "$HF_CLI" upload "$HF_REPO" "$MANIFEST" MANIFEST_final.tsv --repo-type model --commit-message "manifest"
  ;;
verify)
  python3 - "$MANIFEST" "$HF_REPO" <<'PY'
import csv, sys
from huggingface_hub import HfApi
man, repo = sys.argv[1], sys.argv[2]
rows = {r["file"]: r for r in csv.DictReader(open(man), delimiter="\t")}
hub = {f.path: f for f in HfApi().list_repo_tree(repo, recursive=True, expand=True) if hasattr(f, "size")}
bad = []
for name, r in rows.items():
    f = hub.get(name)
    got = f.lfs.sha256 if f is not None and getattr(f, "lfs", None) else None
    if f is None or int(r["size_bytes"]) != f.size or (got is not None and got != r["sha256"]):
        bad.append(name)
print(f"{len(rows)} files in the manifest, {len(rows) - len(bad)} on the Hub with the right size / sha256")
if bad:
    print("missing or different:", *bad[:20], sep="\n  "); sys.exit(1)
PY
  ;;
*) echo "unknown command $cmd" >&2; exit 2 ;;
esac
