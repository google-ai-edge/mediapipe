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
example apps, start from [Hello World! on Android](./hello_world_android.md). To
incorporate MediaPipe into an existing Android Studio project, see these
[instructions](./android_archive_library.md) that use Android Archive (AAR) and
Gradle.

## Building Android example apps

### Prerequisite

*   Install MediaPipe following these [instructions](./install.md).
*   Setup Java Runtime.
*   Setup Android SDK release 28.0.3 and above.
*   Setup Android NDK r18b and above.

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

In order to use MediaPipe on earlier Android versions, MediaPipe needs to switch
to a lower Android API level. You can achieve this by specifying `api_level =
$YOUR_INTENDED_API_LEVEL` in android_ndk_repository() and/or
android_sdk_repository() in the
[`WORKSPACE`](https://github.com/google/mediapipe/blob/master/WORKSPACE) file.

Please verify all the necessary packages are installed.

*   Android SDK Platform API Level 28 or 29
*   Android SDK Build-Tools 28 or 29
*   Android SDK Platform-Tools 28 or 29
*   Android SDK Tools 26.1.1
*   Android NDK 17c or above

### Option 1: Build with Bazel in Command Line

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

### Option 2: Build with Bazel in Android Studio

The MediaPipe project can be imported into Android Studio using the Bazel
plugins. This allows the MediaPipe examples to be built and modified in Android
Studio.

To incorporate MediaPipe into an existing Android Studio project, see these
[instructions](./android_archive_library.md) that use Android Archive (AAR) and
Gradle.

The steps below use Android Studio 3.5 to build and install a MediaPipe example
app:

1.  Install and launch Android Studio 3.5.

2.  Select `Configure` -> `SDK Manager` -> `SDK Platforms`.

    *   Verify that Android SDK Platform API Level 28 or 29 is installed.
    *   Take note of the Android SDK Location, e.g.,
        `/usr/local/home/Android/Sdk`.

3.  Select `Configure` -> `SDK Manager` -> `SDK Tools`.

    *   Verify that Android SDK Build-Tools 28 or 29 is installed.
    *   Verify that Android SDK Platform-Tools 28 or 29 is installed.
    *   Verify that Android SDK Tools 26.1.1 is installed.
    *   Verify that Android NDK 17c or above is installed.
    *   Take note of the Android NDK Location, e.g.,
        `/usr/local/home/Android/Sdk/ndk-bundle` or
        `/usr/local/home/Android/Sdk/ndk/20.0.5594570`.

4.  Set environment variables `$ANDROID_HOME` and `$ANDROID_NDK_HOME` to point
    to the installed SDK and NDK.

    ```bash
    export ANDROID_HOME=/usr/local/home/Android/Sdk

    # If the NDK libraries are installed by a previous version of Android Studio, do
    export ANDROID_NDK_HOME=/usr/local/home/Android/Sdk/ndk-bundle
    # If the NDK libraries are installed by Android Studio 3.5, do
    export ANDROID_NDK_HOME=/usr/local/home/Android/Sdk/ndk/<version number>
    ```

5.  Select `Configure` -> `Plugins` to install `Bazel`.

6.  On Linux, select `File` -> `Settings` -> `Bazel settings`. On macos, select
    `Android Studio` -> `Preferences` -> `Bazel settings`. Then, modify `Bazel
    binary location` to be the same as the output of `$ which bazel`.

7.  Select `Import Bazel Project`.

    *   Select `Workspace`: `/path/to/mediapipe` and select `Next`.
    *   Select `Generate from BUILD file`: `/path/to/mediapipe/BUILD` and select
        `Next`.
    *   Modify `Project View` to be the following and select `Finish`.

    ```
    directories:
      # read project settings, e.g., .bazelrc
      .
      -mediapipe/objc
      -mediapipe/examples/ios

    targets:
      //mediapipe/examples/android/...:all
      //mediapipe/java/...:all

    android_sdk_platform: android-29

    sync_flags:
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
    ```

8.  Select `Bazel` -> `Sync` -> `Sync project with Build files`.

    Note: Even after doing step 4, if you still see the error: `"no such package
    '@androidsdk//': Either the path attribute of android_sdk_repository or the
    ANDROID_HOME environment variable must be set."`, please modify the
    [`WORKSPACE`](https://github.com/google/mediapipe/blob/master/WORKSPACE)
    file to point to your SDK and NDK library locations, as below:

    ```
    android_sdk_repository(
        name = "androidsdk",
        path = "/path/to/android/sdk"
    )

    android_ndk_repository(
        name = "androidndk",
        path = "/path/to/android/ndk"
    )
    ```

9.  Connect an Android device to the workstation.

10. Select `Run...` -> `Edit Configurations...`.

    *   Select `Templates` -> `Bazel Command`.
    *   Enter Target Expression:
        `//mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu:handtrackinggpu`
    *   Enter Bazel command: `mobile-install`.
    *   Enter Bazel flags: `-c opt --config=android_arm64`.
    *   Press the `[+]` button to add the new configuration.
    *   Select `Run` to run the example app on the connected Android device.
