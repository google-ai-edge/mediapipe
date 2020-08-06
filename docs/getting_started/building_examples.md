---
layout: default
title: Building MediaPipe Examples
parent: Getting Started
nav_order: 2
---

# Building MediaPipe Examples
{: .no_toc }

1. TOC
{:toc}
---

## Android

### Prerequisite

*   Java Runtime.
*   Android SDK release 28.0.3 and above.
*   Android NDK r18b and above.

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

## iOS

### Prerequisite

1.  Install [Xcode](https://developer.apple.com/xcode/), then install the
    Command Line Tools using:

    ```bash
    xcode-select --install
    ```

2.  Install [Bazel](https://bazel.build/).

    We recommend using [Homebrew](https://brew.sh/) to get the latest version.

3.  Set Python 3.7 as the default Python version and install the Python "six"
    library. This is needed for TensorFlow.

    ```bash
    pip3 install --user six
    ```

4.  Clone the MediaPipe repository.

    ```bash
    git clone https://github.com/google/mediapipe.git
    ```

### Set up a bundle ID prefix

All iOS apps must have a bundle ID, and you must have a provisioning profile
that lets you install an app with that ID onto your phone. To avoid clashes
between different MediaPipe users, you need to configure a unique prefix for the
bundle IDs of our iOS demo apps.

If you have a custom provisioning profile, see
[Custom provisioning](#custom-provisioning) below.

Otherwise, run this command to generate a unique prefix:

```bash
python3 mediapipe/examples/ios/link_local_profiles.py
```

### Create an Xcode project

This allows you to edit and debug one of the example apps in Xcode. It also
allows you to make use of automatic provisioning (see later section).

1.  We will use a tool called [Tulsi](https://tulsi.bazel.build/) for generating
    Xcode projects from Bazel build configurations.

    ```bash
    # cd out of the mediapipe directory, then:
    git clone https://github.com/bazelbuild/tulsi.git
    cd tulsi
    # remove Xcode version from Tulsi's .bazelrc (see http://github.com/bazelbuild/tulsi#building-and-installing):
    sed -i .orig '/xcode_version/d' .bazelrc
    # build and run Tulsi:
    sh build_and_run.sh
    ```

    This will install `Tulsi.app` inside the `Applications` directory in your
    home directory.

2.  Open `mediapipe/Mediapipe.tulsiproj` using the Tulsi app.

    Tip: If Tulsi displays an error saying "Bazel could not be found", press the
    "Bazel..." button in the Packages tab and select the `bazel` executable in
    your homebrew `/bin/` directory.

3.  Select the MediaPipe config in the Configs tab, then press the Generate
    button below. You will be asked for a location to save the Xcode project.
    Once the project is generated, it will be opened in Xcode.

    If you get an error about bundle IDs, see the
    [previous section](#set-up-a-bundle-id-prefix).

### Set up provisioning

To install applications on an iOS device, you need a provisioning profile. There
are two options:

1.  Automatic provisioning. This allows you to build and install an app to your
    personal device. The provisining profile is managed by Xcode, and has to be
    updated often (it is valid for about a week).

2.  Custom provisioning. This uses a provisioning profile associated with an
    Apple developer account. These profiles have a longer validity period and
    can target multiple devices, but you need a paid developer account with
    Apple to obtain one.

#### Automatic provisioning

1.  Create an Xcode project for MediaPipe, as discussed
    [earlier](#create-an-xcode-project).

2.  In the project navigator in the left sidebar, select the "Mediapipe"
    project.

3.  Select one of the application targets, e.g. HandTrackingGpuApp.

4.  Select the "Signing & Capabilities" tab.

5.  Check "Automatically manage signing", and confirm the dialog box.

6.  Select "_Your Name_ (Personal Team)" in the Team pop-up menu.

7.  This set-up needs to be done once for each application you want to install.
    Repeat steps 3-6 as needed.

This generates provisioning profiles for each app you have selected. Now we need
to tell Bazel to use them. We have provided a script to make this easier.

1.  In the terminal, to the `mediapipe` directory where you cloned the
    repository.

2.  Run this command:

    ```bash
    python3 mediapipe/examples/ios/link_local_profiles.py
    ```

This will find and link the provisioning profile for all applications for which
you have enabled automatic provisioning in Xcode.

Note: once a profile expires, Xcode will generate a new one; you must then run
this script again to link the updated profiles.

#### Custom provisioning

1.  Obtain a provisioning profile from Apple.

Tip: You can use this command to see the provisioning profiles you have
previously downloaded using Xcode: `open ~/Library/MobileDevice/"Provisioning
Profiles"`. If there are none, generate and download a profile on
[Apple's developer site](https://developer.apple.com/account/resources/).

1.  Symlink or copy your provisioning profile to
    `mediapipe/mediapipe/provisioning_profile.mobileprovision`.

    ```bash
    cd mediapipe
    ln -s ~/Downloads/MyProvisioningProfile.mobileprovision mediapipe/provisioning_profile.mobileprovision
    ```

Note: if you had previously set up automatic provisioning, you should remove the
`provisioning_profile.mobileprovision` symlink in each example's directory,
since it will take precedence over the common one. You can also overwrite it
with you own profile if you need a different profile for different apps.

1.  Open `mediapipe/examples/ios/bundle_id.bzl`, and change the
    `BUNDLE_ID_PREFIX` to a prefix associated with your provisioning profile.

### Build and run an app using Xcode

1.  Create the Xcode project, and make sure you have set up either automatic or
    custom provisioning.

2.  You can now select any of the MediaPipe demos in the target menu, and build
    and run them as normal.

Note: When you ask Xcode to run an app, by default it will use the Debug
configuration. Some of our demos are computationally heavy; you may want to use
the Release configuration for better performance.

Tip: To switch build configuration in Xcode, click on the target menu, choose
"Edit Scheme...", select the Run action, and switch the Build Configuration from
Debug to Release. Note that this is set independently for each target.

Tip: On the device, in Settings > General > Device Management, make sure the
developer (yourself) is trusted.

### Build an app using the command line

1.  Make sure you have set up either automatic or custom provisioning.

2.  Using [MediaPipe Hands](../solutions/hands.md) for example, run:

    ```bash
    bazel build -c opt --config=ios_arm64 mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp
    ```

    You may see a permission request from `codesign` in order to sign the app.

    Tip: If you are using custom provisioning, you can run this
    [script](https://github.com/google/mediapipe/blob/master/build_ios_examples.sh)
    to build all MediaPipe iOS example apps.

3.  In Xcode, open the `Devices and Simulators` window (command-shift-2).

4.  Make sure your device is connected. You will see a list of installed apps.
    Press the "+" button under the list, and select the `.ipa` file built by
    Bazel.

5.  You can now run the app on your device.

Tip: On the device, in Settings > General > Device Management, make sure the
developer (yourself) is trusted.

## Desktop

### Option 1: Running on CPU

1.  To build, for example, [MediaPipe Hands](../solutions/hands.md), run:

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
    ```

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible.

### Option 2: Running on GPU

Note: This currently works only on Linux, and please first follow
[OpenGL ES Setup on Linux Desktop](./gpu_support.md#opengl-es-setup-on-linux-desktop).

1.  To build, for example, [MediaPipe Hands](../solutions/hands.md), run:

    ```bash
    bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
      mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu
    ```

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible, or GPU drivers not setup
    properly.
