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

Tip: You can use this command to see the provisioning profiles you have
previously downloaded using Xcode: `open ~/Library/MobileDevice/"Provisioning Profiles"`.
If there are none, generate and download a profile on [Apple's developer site](https://developer.apple.com/account/resources/).

## Creating an Xcode project

Note: This workflow requires a separate tool in addition to Bazel. If it fails
to work for any reason, you can always use the command-line build instructions
in the next section.

1.  We will use a tool called [Tulsi](https://tulsi.bazel.build/) for generating Xcode projects from Bazel
    build configurations.

    IMPORTANT: At the time of this writing, Tulsi has a small [issue](https://github.com/bazelbuild/tulsi/issues/98)
    that keeps it from building with Xcode 10.3. The instructions below apply a
    fix from a [pull request](https://github.com/bazelbuild/tulsi/pull/99).

    ```bash
    # cd out of the mediapipe directory, then:
    git clone https://github.com/bazelbuild/tulsi.git
    cd tulsi
    # Apply the fix for Xcode 10.3 compatibility:
    git fetch origin pull/99/head:xcodefix
    git checkout xcodefix
    # Now we can build Tulsi.
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

Note: When you ask Xcode to run an app, by default it will use the Debug
configuration. Some of our demos are computationally heavy; you may want to use
the Release configuration for better performance.

Tip: To switch build configuration in Xcode, click on the target menu, choose
"Edit Scheme...", select the Run action, and switch the Build Configuration from
Debug to Release. Note that this is set independently for each target.

## Building an iOS app from the command line

1.  Modify the `bundle_id` field of the app's ios_application rule to use your own identifier, e.g. for [Face Detection GPU App example](./face_detection_mobile_gpu.md), you need to modify the line 26 of the [BUILD file](https://github.com/google/mediapipe/blob/master/mediapipe/examples/ios/facedetectiongpu/BUILD).

2.  Build one of the example apps for iOS. We will be using the
    [Face Detection GPU App example](./face_detection_mobile_gpu.md)

    ```bash
    cd mediapipe
    bazel build --config=ios_arm64 mediapipe/examples/ios/facedetectiongpu:FaceDetectionGpuApp
    ```

    You may see a permission request from `codesign` in order to sign the app.

3.  In Xcode, open the `Devices and Simulators` window (command-shift-2).

4.  Make sure your device is connected. You will see a list of installed apps.
    Press the "+" button under the list, and select the `.ipa` file built by
    Bazel.

5.  You can now run the app on your device.
