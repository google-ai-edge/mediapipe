# MediaPipe iOS Examples: Build, Install, and Run

This guide documents how to build and deploy the iOS GPU examples in this repository with Bazelisk on a physical device (arm64).

## Prerequisites

- Xcode and Command Line Tools (latest)
- Bazelisk (`brew install bazelisk`)
- Python 3.10–3.12 (hermetic Python bundled by TensorFlow is used automatically)

## Bundle ID and Provisioning

- Bundle ID prefix: `mediapipe/examples/ios/bundle_id.bzl` (`BUNDLE_ID_PREFIX`)
- Provisioning profile lookup order:
  1) `mediapipe/examples/ios/<app>/provisioning_profile.mobileprovision`
  2) `mediapipe/provisioning_profile.mobileprovision`

Wildcard profiles like `com.yourco.*` work fine as long as the prefix matches `BUNDLE_ID_PREFIX`.

## Device Build (arm64)

```
export USE_BAZEL_VERSION=6.5.0
bazelisk build -c opt --config=ios_arm64 \
  //mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp

# Artifacts:
# bazel-bin/mediapipe/examples/ios/handtrackinggpu/HandTrackingGpuApp.ipa
# bazel-bin/mediapipe/examples/ios/handtrackinggpu/HandTrackingGpuApp.app.dSYM
```

Install and launch (requires a trusted, connected iPhone):

```
UDID=$(xcrun xctrace list devices | sed -n 's/.*(\([0-9A-F-]\{25,\}\)).*/\1/p' | head -n1)
TMPDIR=$(mktemp -d)
unzip -q bazel-bin/mediapipe/examples/ios/handtrackinggpu/HandTrackingGpuApp.ipa -d "$TMPDIR"
xcrun devicectl device install app --device "$UDID" \
  "$TMPDIR/Payload/HandTrackingGpuApp.app"
xcrun devicectl device process launch --device "$UDID" \
  com.codexmp.mediapipe.examples.HandTrackingGpu --console --terminate-existing
```

## Notes
- OpenCV is included for device builds via a prebuilt framework; no special setup is required.

## Holistic Tracking

The same commands apply to the Holistic example:

```
# Device
bazelisk build -c opt --config=ios_arm64 \
  //mediapipe/examples/ios/holistictrackinggpu:HolisticTrackingGpuApp
```

## Troubleshooting

- Provisioning/signing errors on device builds → Update the provisioning profile at one of the documented paths and ensure `BUNDLE_ID_PREFIX` matches the profile's prefix.
- `bazel run` launches simulator instead of device → Use `devicectl` commands above to install/launch directly on a physical device.
