---
layout: forward
target: https://developers.google.com/mediapipe/framework/getting_started/ios
title: MediaPipe on iOS
parent: Getting Started
has_children: true
has_toc: false
nav_order: 2
---

# MediaPipe on iOS
{: .no_toc }

1. TOC
{:toc}
---

**Attention:** *Thanks for your interest in MediaPipe! We have moved to
[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
as the primary developer documentation site for MediaPipe as of April 3, 2023.*

----

Please follow instructions below to build iOS example apps in the supported
MediaPipe [solutions](../solutions/solutions.md). To learn more about these
example apps, start from, start from
[Hello World! on iOS](./hello_world_ios.md).

## Building iOS example apps

### Prerequisite

1.  Install MediaPipe following these [instructions](./install.md).

2.  Install [Xcode](https://developer.apple.com/xcode/), then install the
    Command Line Tools using:

    ```bash
    xcode-select --install
    ```

3.  Install [Bazelisk](https://github.com/bazelbuild/bazelisk)
.

    We recommend using [Homebrew](https://brew.sh/) to get the latest versions.

    ```bash
    brew install bazelisk
    ```

4.  Set Python 3.7 as the default Python version and install the Python "six"
    library. This is needed for TensorFlow.

    ```bash
    pip3 install --user six
    ```

5.  Clone the MediaPipe repository.

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

    **Note**: Please ensure the `xcode_version` in the
    [`build_and_run.sh`](https://github.com/bazelbuild/tulsi/blob/b1d0108e6a93dbe8ab01529b2c607b6b651f0759/build_and_run.sh#L26)
    file in tulsi repo is the same version as installed in your system.

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

1.  Automatic provisioning. This allows you to build and install an app on your
    personal device. The provisioning profile is managed by Xcode, and has to be
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
with your own profile if you need a different profile for different apps.

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

Note: Due to an incompatibility caused by one of our dependencies, MediaPipe
cannot be used for apps running on the iPhone Simulator on Apple Silicon (M1).

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
