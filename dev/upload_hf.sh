#!/usr/bin/env bash
# Upload the staged S2V bundles to Hugging Face, one repository at a time.
#
#     dev/upload_hf.sh                # all four, Lite first
#     dev/upload_hf.sh Lite           # only Kandinsky-S2V-1.0-Lite
#     dev/upload_hf.sh Pro Pro-distill
#
# The token is read from $HF_TOKEN or prompted for. It needs write access to
# the kandinskylab organisation itself — a token scoped only to the user
# account sees the org as if it did not exist and fails on repo creation.
#
# Bundles are staged under $BUNDLES; the weights there are symlinks into
# weights_k6_s2v/, which `hf upload` follows.
set -euo pipefail

ORG=kandinskylab
# Override to upload a different set, e.g. the Diffusers-format bundles:
#   BUNDLES=.../bundles_diffusers dev/upload_hf.sh Lite-Diffusers
BUNDLES=${BUNDLES:-/home/jovyan/shares/SR008.fs2/semenov/weights_k6_s2v/bundles}
HF="$(dirname "$0")/../.venv/bin/hf"

repos=(
    Kandinsky-S2V-1.0-Lite
    Kandinsky-S2V-1.0-Lite-distill
    Kandinsky-S2V-1.0-Pro
    Kandinsky-S2V-1.0-Pro-distill
)
if [ $# -gt 0 ]; then
    repos=()
    for name in "$@"; do
        case $name in
            Kandinsky-*) repos+=("$name") ;;
            *) repos+=("Kandinsky-S2V-1.0-$name") ;;
        esac
    done
fi

for repo in "${repos[@]}"; do
    [ -d "$BUNDLES/$repo" ] || { echo "no bundle staged at $BUNDLES/$repo" >&2; exit 1; }
done

if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "HF token: " HF_TOKEN
    echo
    export HF_TOKEN
fi

"$HF" auth whoami

for repo in "${repos[@]}"; do
    dir=$BUNDLES/$repo
    echo
    echo "=== $ORG/$repo ($(du -Lsh "$dir" | cut -f1)) ==="

    "$HF" repo create "$ORG/$repo" --type model --private --exist-ok
    "$HF" upload "$ORG/$repo" "$dir" . \
        --type model --private --exclude ".cache/**" \
        --commit-message "$repo"
    "$HF" models ls -R "$ORG/$repo"
done
