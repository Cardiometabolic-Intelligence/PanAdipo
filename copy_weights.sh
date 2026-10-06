#!/usr/bin/env bash
# Stage the fine-tuned PanAdipo checkpoint into this repo's weights/ dir.
#
# By default, downloads best_model.pt from the GitHub Release. Optionally copy
# from a local file instead, by passing a path or setting PANADIPO_SRC:
#   ./copy_weights.sh                         # download from the release (default)
#   ./copy_weights.sh /path/to/best_model.pt  # copy from a local file
#   PANADIPO_SRC=/path/to/best_model.pt ./copy_weights.sh
#
# For a public repo no auth is needed. For a private repo, provide auth via the
# `gh` CLI (gh auth login) or a token:  export GH_TOKEN=ghp_xxx
set -euo pipefail

REPO="Cardiometabolic-Intelligence/PanAdipo"
TAG="v1.0"
ASSET="best_model.pt"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$REPO_DIR/weights"
DEST="$DEST_DIR/$ASSET"

# Optional local source: CLI arg takes precedence, else PANADIPO_SRC env var.
SRC="${1:-${PANADIPO_SRC:-}}"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST" ]]; then
  echo "Already present: $DEST  ($(du -h "$DEST" | cut -f1))"
  echo "Delete it and re-run to refresh."
  exit 0
fi

if [[ -n "$SRC" ]]; then
  if [[ ! -f "$SRC" ]]; then
    echo "ERROR: local source not found: $SRC" >&2
    exit 1
  fi
  echo "Copying from local source:"
  echo "  $SRC -> $DEST"
  cp -v "$SRC" "$DEST"
else
  echo "Downloading $ASSET from GitHub Release $TAG …"
  if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir "$DEST_DIR"
  elif [[ -n "${GH_TOKEN:-}" ]]; then
    echo "  Using GH_TOKEN via API …"
    asset_id=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" \
      "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
      | python -c 'import sys,json,os;d=json.load(sys.stdin);print(next(a["id"] for a in d["assets"] if a["name"]==os.environ["ASSET"]))')
    curl -sfL -H "Authorization: Bearer $GH_TOKEN" \
      -H "Accept: application/octet-stream" \
      "https://api.github.com/repos/$REPO/releases/assets/$asset_id" \
      -o "$DEST"
  else
    echo "ERROR: could not download (no gh CLI, no GH_TOKEN)." >&2
    echo "Install gh, OR set GH_TOKEN, OR pass a local path." >&2
    echo "(Or pass a local path: ./copy_weights.sh /path/to/best_model.pt)" >&2
    exit 1
  fi
fi

echo "Done ($(du -h "$DEST" | cut -f1))."

python - "$DEST" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
keys = list(ck.get("model_state", {}).keys())
print(f"  model_state tensors : {len(keys)}")
print(f"  head type           : {'4-layer' if 'head.4.weight' in keys else 'simple'}")
print(f"  clip_len            : {(ck.get('args') or {}).get('clip_len', 16)}")
import hashlib
md5 = hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest()
ok = md5 == "1aa0ce1989506553dca91f3982a199cf"
print(f"  md5                 : {md5}  ({'OK' if ok else 'MISMATCH - re-download'})")
PY
