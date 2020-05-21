# Building MediaPipe Examples

*   [Android](#android)
*   [iOS](#ios)
*   [Desktop](#desktop)

## Android

### Prerequisite

*   Java Runtime.
*   Android SDK release 28.0.3 and above.
*   Android NDK r18b and above.

MediaPipe recommends setting up Android SDK and NDK via Android Studio (and see
below for Android Studio setup). However, if you prefer using MediaPipe without
Android Studio, please run
[`setup_android_sdk_and_ndk.sh`](https://github.com/google/mediapipe/tree/master/setup_android_sdk_and_ndk.sh)
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
<api level integer>` in android_ndk_repository() and/or android_sdk_repository()
in the [`WORKSPACE`](https://github.com/google/mediapipe/tree/master/WORKSPACE) file.

Please verify all the necessary packages are installed.

*   Android SDK Platform API Level 28 or 29
*   Android SDK Build-Tools 28 or 29
*   Android SDK Platform-Tools 28 or 29
*   Android SDK Tools 26.1.1
*   Android NDK 17c or above

### Option 1: Build with Bazel in Command Line

1.  To build an Android example app, for instance, for MediaPipe Hand, run:

    Note: To reduce the binary size, consider appending `--linkopt="-s"` to the
    command below to strip symbols.

    ~~~
      ```bash
    bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu
    ```
    ~~~

1.  Install it on a device with:

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

2.  Select `Configure` | `SDK Manager` | `SDK Platforms`.

    *   Verify that Android SDK Platform API Level 28 or 29 is installed.
    *   Take note of the Android SDK Location, e.g.,
        `/usr/local/home/Android/Sdk`.

3.  Select `Configure` | `SDK Manager` | `SDK Tools`.

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

5.  Select `Configure` | `Plugins` install `Bazel`.

6.  On Linux, select `File` | `Settings`| `Bazel settings`. On macos, select
    `Android Studio` | `Preferences` | `Bazel settings`. Then, modify `Bazel
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

8.  Select `Bazel` | `Sync` | `Sync project with Build files`.

    Note: Even after doing step 4, if you still see the error: `"no such package
    '@androidsdk//': Either the path attribute of android_sdk_repository or the
    ANDROID_HOME environment variable must be set."`, please modify the
    [`WORKSPACE`](https://github.com/google/mediapipe/tree/master/WORKSPACE) file to point to your
    SDK and NDK library locations, as below:

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

10. Select `Run...` | `Edit Configurations...`.

    *   Select `Templates` | `Bazel Command`.
    *   Enter Target Expression:
        `//mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu`
    *   Enter Bazel command: `mobile-install`.
    *   Enter Bazel flags: `-c opt --config=android_arm64`.
    *   Press the `[+]` button to add the new configuration.
    *   Select `Run` to run the example app on the connected Android device.

## iOS

### Prerequisite

1.  Install [Xcode](https://developer.apple.com/xcode/) and the Command Line
    Tools.

    Follow Apple's instructions to obtain the required development certificates
    and provisioning profiles for your iOS device. Install the Command Line
    Tools by

    ```bash
    xcode-select --install
    ```

2.  Install [Bazel](https://bazel.build/).

    We recommend using [Homebrew](https://brew.sh/) to get the latest version.

3.  Set Python 3.7 as the default Python version and install the Python "six"
    library.

    To make Mediapipe work with TensorFlow, please set Python 3.7 as the default
    Python version and install the Python "six" library.

    ```bash
    pip3 install --user six
    ```

4.  Clone the MediaPipe repository.

    ```bash
    git clone https://github.com/google/mediapipe.git
    ```

5.  Symlink or copy your provisioning profile to
    `mediapipe/mediapipe/provisioning_profile.mobileprovision`.

    ```bash
    cd mediapipe
    ln -s ~/Downloads/MyProvisioningProfile.mobileprovision mediapipe/provisioning_profile.mobileprovision
    ```

    Tip: You can use this command to see the provisioning profiles you have
    previously downloaded using Xcode: `open
    ~/Library/MobileDevice/"Provisioning Profiles"`. If there are none, generate
    and download a profile on
    [Apple's developer site](https://developer.apple.com/account/resources/).

### Option 1: Build with Bazel in Command Line

1.  Modify the `bundle_id` field of the app's `ios_application` target to use
    your own identifier. For instance, for
    [MediaPipe Hand](./hand_tracking_mobile_gpu.md), the `bundle_id` is in the
    `HandTrackingGpuApp` target in the
    [BUILD](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/handtrackinggpu/BUILD)
    file.

2.  Again using [MediaPipe Hand](./hand_tracking_mobile_gpu.md) for example,
    run:

    ```bash
    bazel build -c opt --config=ios_arm64 mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp
    ```

    You may see a permission request from `codesign` in order to sign the app.

3.  In Xcode, open the `Devices and Simulators` window (command-shift-2).

4.  Make sure your device is connected. You will see a list of installed apps.
    Press the "+" button under the list, and select the `.ipa` file built by
    Bazel.

5.  You can now run the app on your device.

### Option 2: Build in Xcode

Note: This workflow requires a separate tool in addition to Bazel. If it fails
to work for some reason, please resort to the command-line build instructions in
the previous section.

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

    Important: If Tulsi displays an error saying "Bazel could not be found",
    press the "Bazel..." button in the Packages tab and select the `bazel`
    executable in your homebrew `/bin/` directory.

3.  Select the MediaPipe config in the Configs tab, then press the Generate
    button below. You will be asked for a location to save the Xcode project.
    Once the project is generated, it will be opened in Xcode.

4.  You can now select any of the MediaPipe demos in the target menu, and build
    and run them as normal.

    Note: When you ask Xcode to run an app, by default it will use the Debug
    configuration. Some of our demos are computationally heavy; you may want to
    use the Release configuration for better performance.

    Tip: To switch build configuration in Xcode, click on the target menu,
    choose "Edit Scheme...", select the Run action, and switch the Build
    Configuration from Debug to Release. Note that this is set independently for
    each target.

## Desktop

### Option 1: Running on CPU

1.  To build, for example, [MediaPipe Hand](./hand_tracking_mobile_gpu.md), run:

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible.

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
    ```

### Option 2: Running on GPU

Note: This currently works only on Linux, and please first follow
[OpenGL ES Setup on Linux Desktop](./gpu.md#opengl-es-setup-on-linux-desktop).

1.  To build, for example, [MediaPipe Hand](./hand_tracking_mobile_gpu.md), run:

    ```bash
    bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
      mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible, or GPU drivers not setup
    properly.

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt
    ```
