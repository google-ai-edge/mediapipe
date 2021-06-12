#!/bin/sh

set -e
set -v

echo 'Please run this from root level mediapipe directory! \n Ex:'
echo '  sh mediapipe/examples/coral/setup.sh  '

sleep 3

mkdir -p opencv32_arm64_libs

# prepare docker aux script
cp mediapipe/examples/coral/update_sources.sh update_sources.sh
chmod +x update_sources.sh

# backup non-coral Dockerfile
mv Dockerfile Dockerfile.orig
cp mediapipe/examples/coral/Dockerfile Dockerfile

# backup non-coral workspace
cp WORKSPACE WORKSPACE.orig

# create temps
cp WORKSPACE WORKSPACE.1
cp mediapipe/examples/coral/WORKSPACE.coral WORKSPACE.2

# merge (shell decides concat order, unless numbered appropriately)
cat WORKSPACE.1 WORKSPACE.2 > WORKSPACE

# cleanup
rm WORKSPACE.1 WORKSPACE.2

echo 'done'
