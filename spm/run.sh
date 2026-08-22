#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MPP_ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$MPP_ROOT_DIR"

rm -rf spm/output

"$SCRIPT_DIR/build.sh"

"$SCRIPT_DIR/generate-package-swift.sh"

git add -A

git commit -m "Update Package.swift"

git push -f

"$SCRIPT_DIR/upload-release.sh"

gh release edit "v${MPP_BUILD_VERSION}" --repo "${GITHUB_REPO}" --draft=false
