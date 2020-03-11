#!/bin/sh

set -e
set -v

echo 'Please run this from root level mediapipe directory! \n Ex:'
echo '  sh mediapipe/examples/coral/setup.sh  '

sleep 3

mkdir -p opencv32_arm64_libs

cp mediapipe/examples/coral/update_sources.sh update_sources.sh
chmod +x update_sources.sh

mv Dockerfile Dockerfile.orig
cp mediapipe/examples/coral/Dockerfile Dockerfile

cp WORKSPACE WORKSPACE.orig
cp mediapipe/examples/coral/WORKSPACE WORKSPACE

