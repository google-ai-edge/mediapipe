## Setting up MediaPipe for iOS

1.  Install [Xcode](https://developer.apple.com/xcode/).

    Follow Apple's instructions to obtain the required developemnt certificates
    and provisioning profiles for your iOS device.

2.  Install [Bazel](https://bazel.build/).

    See their [instructions](https://docs.bazel.build/versions/master/install-os-x.html).
    We recommend using [Homebrew](https://brew.sh/):

    ```bash
    brew tap bazelbuild/tap
    brew install bazelbuild/tap/bazel
    ```

3.  Clone the MediaPipe repository.

    ```bash
    git clone https://github.com/google/mediapipe.git
    ```

4.  Symlink or copy your provisioning profile to `mediapipe/mediapipe/provisioning_profile.mobileprovision`.

    ```bash
    cd mediapipe
    ln -s ~/Downloads/MyProvisioningProfile.mobileprovision mediapipe/provisioning_profile.mobileprovision
    ```

## Creating an Xcode project

1.  We will use a tool called [Tulsi](https://tulsi.bazel.build/) for generating Xcode projects from Bazel
    build configurations.

    ```bash
    git clone https://github.com/bazelbuild/tulsi.git
    cd tulsi
    sh build_and_run.sh
    ```

    This will install Tulsi.app inside the Applications directory inside your
    home directory.

2.  Open `mediapipe/Mediapipe.tulsiproj` using the Tulsi app.

3.  Select the MediaPipe config in the Configs tab, then press the Generate
    button below. You will be asked for a location to save the Xcode project.
    Once the project is generated, it will be opened in Xcode.

4.  You can now select any of the MediaPipe demos in the target menu, and build
    and run them as normal.

## Building an iOS app from the command line

1.  Build one of the example apps for iOS. We will be using the
    [Face Detection GPU App example](./face_detection_mobile_gpu.md)

    ```bash
    bazel build --config=ios_arm64 mediapipe/examples/ios/facedetectiongpu:FaceDetectionGpuApp
    ```

    You may see a permission request from `codesign` in order to sign the app.

2.  In Xcode, open the `Devices and Simulators` window (command-shift-2).

3.  Make sure your device is connected. You will see a list of installed apps.
    Press the "+" button under the list, and select the `.ipa` file built by
    Bazel.

4.  You can now run the app on your device.
