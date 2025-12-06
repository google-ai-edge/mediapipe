#!/usr/bin/env bash
set -e

rm -rf output

./build.sh

./generate-package-swift.sh

git add -A

git commit -m "Update Package.swift"

git push

./upload-release.sh

gh release edit v0.10.26 --repo mihaidimoiu/mediapipe --draft=false
