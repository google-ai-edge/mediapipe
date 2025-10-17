---
layout: forward
target: https://developers.google.com/mediapipe
title: Home
nav_order: 1
---

----

**Attention:** *We have moved to
[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
as the primary developer documentation site for MediaPipe as of April 3, 2023.*

![MediaPipe](https://developers.google.com/static/mediapipe/images/home/hero_01_1920.png)

**Attention**: MediaPipe Solutions Preview is an early release. [Learn
more](https://developers.google.com/mediapipe/solutions/about#notice).

**On-device machine learning for everyone**

Delight your customers with innovative machine learning features. MediaPipe
contains everything that you need to customize and deploy to mobile (Android,
iOS), web, desktop, edge devices, and IoT, effortlessly.

*   [See demos](https://goo.gle/mediapipe-studio)
*   [Learn more](https://developers.google.com/mediapipe/solutions)

## Get started

You can get started with MediaPipe Solutions by by checking out any of the
developer guides for
[vision](https://developers.google.com/mediapipe/solutions/vision/object_detector),
[text](https://developers.google.com/mediapipe/solutions/text/text_classifier),
and
[audio](https://developers.google.com/mediapipe/solutions/audio/audio_classifier)
tasks. If you need help setting up a development environment for use with
MediaPipe Tasks, check out the setup guides for
[Android](https://developers.google.com/mediapipe/solutions/setup_android), [web
apps](https://developers.google.com/mediapipe/solutions/setup_web), and
[Python](https://developers.google.com/mediapipe/solutions/setup_python).

## Solutions

MediaPipe Solutions provides a suite of libraries and tools for you to quickly
apply artificial intelligence (AI) and machine learning (ML) techniques in your
applications. You can plug these solutions into your applications immediately,
customize them to your needs, and use them across multiple development
platforms. MediaPipe Solutions is part of the MediaPipe [open source
project](https://github.com/google/mediapipe), so you can further customize the
solutions code to meet your application needs.

These libraries and resources provide the core functionality for each MediaPipe
Solution:

*   **MediaPipe Tasks**: Cross-platform APIs and libraries for deploying
    solutions. [Learn
    more](https://developers.google.com/mediapipe/solutions/tasks).
*   **MediaPipe models**: Pre-trained, ready-to-run models for use with each
    solution.

These tools let you customize and evaluate solutions:

*   **MediaPipe Model Maker**: Customize models for solutions with your data.
    [Learn more](https://developers.google.com/mediapipe/solutions/model_maker).
*   **MediaPipe Studio**: Visualize, evaluate, and benchmark solutions in your
    browser. [Learn
    more](https://developers.google.com/mediapipe/solutions/studio).

### Legacy solutions

We have ended support for [these MediaPipe Legacy Solutions](https://developers.google.com/mediapipe/solutions/guide#legacy)
as of March 1, 2023. All other MediaPipe Legacy Solutions will be upgraded to
a new MediaPipe Solution. See the [Solutions guide](https://developers.google.com/mediapipe/solutions/guide#legacy)
for details. The [code repository](https://github.com/google/mediapipe/tree/master/mediapipe)
and prebuilt binaries for all MediaPipe Legacy Solutions will continue to be
provided on an as-is basis.

For more on the legacy solutions, see the [documentation](https://github.com/google/mediapipe/tree/master/docs/solutions).

## Framework

To start using MediaPipe Framework, [install MediaPipe
Framework](https://developers.google.com/mediapipe/framework/getting_started/install)
and start building example applications in C++, Android, and iOS.

[MediaPipe Framework](https://developers.google.com/mediapipe/framework) is the
low-level component used to build efficient on-device machine learning
pipelines, similar to the premade MediaPipe Solutions.

Before using MediaPipe Framework, familiarize yourself with the following key
[Framework
concepts](https://developers.google.com/mediapipe/framework/framework_concepts/overview.md):

*   [Packets](https://developers.google.com/mediapipe/framework/framework_concepts/packets.md)
*   [Graphs](https://developers.google.com/mediapipe/framework/framework_concepts/graphs.md)
*   [Calculators](https://developers.google.com/mediapipe/framework/framework_concepts/calculators.md)

## Community

*   [Slack community](https://mediapipe.page.link/joinslack) for MediaPipe
    users.
*   [Discuss](https://groups.google.com/forum/#!forum/mediapipe) - General
    community discussion around MediaPipe.
*   [Awesome MediaPipe](https://mediapipe.page.link/awesome-mediapipe) - A
    curated list of awesome MediaPipe related frameworks, libraries and
    software.

## Run on Apple Silicon (macOS)

The following quick-start shows how to build and run a simple MediaPipe example
on an Apple Silicon (arm64) Mac. These steps use native arm64 tooling.

- Prerequisites: Xcode + CLT, Homebrew, Bazelisk
  - Install Xcode CLT: `xcode-select --install`
  - Install Homebrew: https://brew.sh
  - Install tools: `brew install bazelisk cmake ninja protobuf opencv pkg-config python`

- Clone your fork and enter the repo
  - `git clone https://github.com/anooshm/mediapipe.git`
  - `cd mediapipe`

- Build and run the desktop Hello World (CPU)
  - Build: `bazelisk build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/examples/desktop/hello_world:hello_world`
  - Run: `bazelisk run -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/examples/desktop/hello_world:hello_world`

Notes
- CPU-only avoids GPU/GL driver differences on macOS; add GPU back later if needed.
- If Homebrew installs OpenCV to a non-standard path, `pkg-config` (installed above)
  typically resolves includes and libs automatically.
- Ensure you are using Apple Silicon-native Homebrew (no Rosetta) for consistent arm64 builds.

For iOS (device) quick start, see the iOS section below.

## Examples (Bazel)

This fork preserves the classic MediaPipe examples. Below are concise Bazel
commands to build and run the iOS GPU examples on a physical device.

### Hand Tracking — Desktop (CPU)

Build and run the desktop CPU example with your Mac’s webcam:

```bash
# Build
bazelisk build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu

# Run (opens default webcam)
GLOG_logtostderr=1 \
  bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
```

Notes:
- If OpenCV is not detected, install: `brew install opencv pkg-config`.
- Grant Terminal camera permissions on first run (macOS privacy prompt).

### Hand Tracking — Desktop (GPU)

Desktop GPU example is supported on Linux per upstream docs. On macOS, prefer:
- Desktop CPU example above, or
- iOS HandTrackingGpu below (runs on-device using Apple GPU/Metal).

### Hand Tracking — iOS (GPU)

### Hand Tracking — iOS (GPU)

Build the iOS app with Bazelisk. You need a valid bundle ID and provisioning
profile (wildcard profiles like `com.yourco.*` are fine).

- Bundle ID prefix: `mediapipe/examples/ios/bundle_id.bzl` (`BUNDLE_ID_PREFIX`)
- Provisioning profile search order:
  - Per‑app: `mediapipe/examples/ios/<app>/provisioning_profile.mobileprovision`
  - Fallback: `mediapipe/provisioning_profile.mobileprovision`

Device (arm64):

```bash
# Pin Bazel if needed
export USE_BAZEL_VERSION=6.5.0

# Build device app (codesigns using your provisioning profile)
bazelisk build -c opt --config=ios_arm64 \
  //mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp

# Resulting artifacts
ls bazel-bin/mediapipe/examples/ios/handtrackinggpu/
# -> HandTrackingGpuApp.ipa, HandTrackingGpuApp.app.dSYM
```

Install and launch on a connected device using Xcode tooling:

```bash
# Find device UDID
xcrun xctrace list devices | sed -n 's/.*(\([0-9A-F-]\{25,\}\)).*/\1/p' | head -n1

# Install the .app from the .ipa payload
TMPDIR=$(mktemp -d)
unzip -q bazel-bin/mediapipe/examples/ios/handtrackinggpu/HandTrackingGpuApp.ipa -d "$TMPDIR"
xcrun devicectl device install app --device <UDID> \
  "$TMPDIR/Payload/HandTrackingGpuApp.app"

# Launch
xcrun devicectl device process launch --device <UDID> \
  com.codexmp.mediapipe.examples.HandTrackingGpu --console --terminate-existing
```

Troubleshooting (device):
- Provisioning errors: update the provisioning profile at one of the paths above and
  ensure the bundle ID prefix matches your profile.

## iOS (Device) — Holistic Tracking (GPU)

Build, install and launch similarly to Hand Tracking:

```bash
export USE_BAZEL_VERSION=6.5.0

# Build (device)
bazelisk build -c opt --config=ios_arm64 \
  //mediapipe/examples/ios/holistictrackinggpu:HolisticTrackingGpuApp

# Install
UDID=$(xcrun xctrace list devices | sed -n 's/.*(\([0-9A-F-]\{25,\}\)).*/\1/p' | head -n1)
TMPDIR=$(mktemp -d)
unzip -q bazel-bin/mediapipe/examples/ios/holistictrackinggpu/HolisticTrackingGpuApp.ipa -d "$TMPDIR"
xcrun devicectl device install app --device "$UDID" \
  "$TMPDIR/Payload/HolisticTrackingGpuApp.app"

# Launch
xcrun devicectl device process launch --device "$UDID" \
  com.codexmp.mediapipe.examples.HolisticTrackingGpu --console --terminate-existing
```

Notes:
- OpenCV is bundled for device builds; no extra setup is required.
- Ensure your provisioning profile covers the Bundle ID prefix set in `bundle_id.bzl`.

## Contributing

We welcome contributions. Please follow these
[guidelines](https://github.com/google/mediapipe/blob/master/CONTRIBUTING.md).

We use GitHub issues for tracking requests and bugs. Please post questions to
the MediaPipe Stack Overflow with a `mediapipe` tag.

## Resources

### Publications

*   [Bringing artworks to life with AR](https://developers.googleblog.com/2021/07/bringing-artworks-to-life-with-ar.html)
    in Google Developers Blog
*   [Prosthesis control via Mirru App using MediaPipe hand tracking](https://developers.googleblog.com/2021/05/control-your-mirru-prosthesis-with-mediapipe-hand-tracking.html)
    in Google Developers Blog
*   [SignAll SDK: Sign language interface using MediaPipe is now available for
    developers](https://developers.googleblog.com/2021/04/signall-sdk-sign-language-interface-using-mediapipe-now-available.html)
    in Google Developers Blog
*   [MediaPipe Holistic - Simultaneous Face, Hand and Pose Prediction, on
    Device](https://ai.googleblog.com/2020/12/mediapipe-holistic-simultaneous-face.html)
    in Google AI Blog
*   [Background Features in Google Meet, Powered by Web ML](https://ai.googleblog.com/2020/10/background-features-in-google-meet.html)
    in Google AI Blog
*   [MediaPipe 3D Face Transform](https://developers.googleblog.com/2020/09/mediapipe-3d-face-transform.html)
    in Google Developers Blog
*   [Instant Motion Tracking With MediaPipe](https://developers.googleblog.com/2020/08/instant-motion-tracking-with-mediapipe.html)
    in Google Developers Blog
*   [BlazePose - On-device Real-time Body Pose Tracking](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html)
    in Google AI Blog
*   [MediaPipe Iris: Real-time Eye Tracking and Depth Estimation](https://ai.googleblog.com/2020/08/mediapipe-iris-real-time-iris-tracking.html)
    in Google AI Blog
*   [MediaPipe KNIFT: Template-based feature matching](https://developers.googleblog.com/2020/04/mediapipe-knift-template-based-feature-matching.html)
    in Google Developers Blog
*   [Alfred Camera: Smart camera features using MediaPipe](https://developers.googleblog.com/2020/03/alfred-camera-smart-camera-features-using-mediapipe.html)
    in Google Developers Blog
*   [Real-Time 3D Object Detection on Mobile Devices with MediaPipe](https://ai.googleblog.com/2020/03/real-time-3d-object-detection-on-mobile.html)
    in Google AI Blog
*   [AutoFlip: An Open Source Framework for Intelligent Video Reframing](https://ai.googleblog.com/2020/02/autoflip-open-source-framework-for.html)
    in Google AI Blog
*   [MediaPipe on the Web](https://developers.googleblog.com/2020/01/mediapipe-on-web.html)
    in Google Developers Blog
*   [Object Detection and Tracking using MediaPipe](https://developers.googleblog.com/2019/12/object-detection-and-tracking-using-mediapipe.html)
    in Google Developers Blog
*   [On-Device, Real-Time Hand Tracking with MediaPipe](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)
    in Google AI Blog
*   [MediaPipe: A Framework for Building Perception Pipelines](https://arxiv.org/abs/1906.08172)

### Videos

*   [YouTube Channel](https://www.youtube.com/c/MediaPipe)
