# Working With Codex CLI in This Repo

This repository works well with the Codex CLI (an open‑source, terminal‑based coding assistant). This document explains the conventions and quick commands the agent can use to help you build and run MediaPipe examples, especially for iOS.

## Conventions

- Preferred build tool: `bazelisk` (pin version via `USE_BAZEL_VERSION=6.5.0`).
- iOS config used by the agent (device only): `--config=ios_arm64`
- Provisioning profile lookup order for device builds:
  1) `mediapipe/examples/ios/<app>/provisioning_profile.mobileprovision`
  2) `mediapipe/provisioning_profile.mobileprovision`
- Bundle ID prefix lives at `mediapipe/examples/ios/bundle_id.bzl`.

## Typical Tasks

- Build HandTrackingGpu for device:
  ```bash
  export USE_BAZEL_VERSION=6.5.0
  bazelisk build -c opt --config=ios_arm64 \
    //mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp
  ```

- Install and launch on a connected iPhone:
  ```bash
  UDID=$(xcrun xctrace list devices | sed -n 's/.*(\([0-9A-F-]\{25,\}\)).*/\1/p' | head -n1)
  TMPDIR=$(mktemp -d)
  unzip -q bazel-bin/mediapipe/examples/ios/handtrackinggpu/HandTrackingGpuApp.ipa -d "$TMPDIR"
  xcrun devicectl device install app --device "$UDID" \
    "$TMPDIR/Payload/HandTrackingGpuApp.app"
  xcrun devicectl device process launch --device "$UDID" \
    com.codexmp.mediapipe.examples.HandTrackingGpu --console --terminate-existing
  ```

## Agent Notes

- The agent may request escalated permissions when invoking Bazel to access Xcode toolchains and external caches.
- If device signing fails, the agent will prompt for an updated provisioning profile matching `BUNDLE_ID_PREFIX`.

For additional details, see `docs/ios_examples.md`.
