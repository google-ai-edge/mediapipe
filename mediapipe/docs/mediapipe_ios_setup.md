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
