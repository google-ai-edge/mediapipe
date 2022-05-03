---
layout: default
title: MediaPipe on Android
parent: Getting Started
has_children: true
has_toc: false
nav_order: 1
---

# MediaPipe on Android
{: .no_toc }

1. TOC
{:toc}
---

Please follow instructions below to build Android example apps in the supported
MediaPipe [solutions](../solutions/solutions.md). To learn more about these
example apps, start from [Hello World! on Android](./hello_world_android.md).

To incorporate MediaPipe into Android Studio projects, see these
[instructions](./android_solutions.md) to use the MediaPipe Android Solution
APIs (currently in alpha) that are now available in
[Google's Maven Repository](https://maven.google.com/web/index.html?#com.google.mediapipe).

## Building Android example apps with Bazel

### Prerequisite

*   Install MediaPipe following these [instructions](./install.md).
*   Setup Java Runtime.
*   Setup Android SDK release 30.0.0 and above.
*   Setup Android NDK version between 18 and 21.

MediaPipe recommends setting up Android SDK and NDK via Android Studio (and see
below for Android Studio setup). However, if you prefer using MediaPipe without
Android Studio, please run
[`setup_android_sdk_and_ndk.sh`](https://github.com/google/mediapipe/blob/master/setup_android_sdk_and_ndk.sh)
to download and setup Android SDK and NDK before building any Android example
apps.

If Android SDK and NDK are already installed (e.g., by Android Studio), set
$ANDROID_HOME and $ANDROID_NDK_HOME to point to the installed SDK and NDK.

```bash
export ANDROID_HOME=<path to the Android SDK>
export ANDROID_NDK_HOME=<path to the Android NDK>
```

and add android_ndk_repository() and android_sdk_repository() rules into the
[`WORKSPACE`](https://github.com/google/mediapipe/blob/master/WORKSPACE) file as
the following:

```bash
$ echo "android_sdk_repository(name = \"androidsdk\")" >> WORKSPACE
$ echo "android_ndk_repository(name = \"androidndk\", api_level=21)" >> WORKSPACE
```

In order to use MediaPipe on earlier Android versions, MediaPipe needs to switch
to a lower Android API level. You can achieve this by specifying `api_level =
$YOUR_INTENDED_API_LEVEL` in android_ndk_repository() and/or
android_sdk_repository() in the
[`WORKSPACE`](https://github.com/google/mediapipe/blob/master/WORKSPACE) file.

Tip: You can run this
[script](https://github.com/google/mediapipe/blob/master/build_android_examples.sh)
to build (and install) all MediaPipe Android example apps.

1.  To build an Android example app, build against the corresponding
    `android_binary` build target. For instance, for
    [MediaPipe Hands](../solutions/hands.md) the target is `handtrackinggpu` in
    the
    [BUILD](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu/BUILD)
    file:

    Note: To reduce the binary size, consider appending `--linkopt="-s"` to the
    command below to strip symbols.

    ```bash
    bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu:handtrackinggpu
    ```

2.  Install it on a device with:

    ```bash
    adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu/handtrackinggpu.apk
    ```
