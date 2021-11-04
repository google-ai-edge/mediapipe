---
layout: default
title: MediaPipe Android Solutions
parent: MediaPipe on Android
grand_parent: Getting Started
nav_order: 2
---

# MediaPipe Android Solutions
{: .no_toc }

1. TOC
{:toc}
---

MediaPipe Android Solution APIs (currently in alpha) are available in:

*   [MediaPipe Face Detection](../solutions/face_detection#android-solution-api)
*   [MediaPipe Face Mesh](../solutions/face_mesh#android-solution-api)
*   [MediaPipe Hands](../solutions/hands#android-solution-api)

## Incorporation in Android Studio

Prebuilt packages of Android Solution APIs can be found in
[Google's Maven Repository](https://maven.google.com/web/index.html?#com.google.mediapipe).
To incorporate them into an Android Studio project, add the following into the
project's Gradle dependencies:

```
dependencies {
    // MediaPipe solution-core is the foundation of any MediaPipe Solutions.
    implementation 'com.google.mediapipe:solution-core:latest.release'
    // Optional: MediaPipe Face Detection Solution.
    implementation 'com.google.mediapipe:facedetection:latest.release'
    // Optional: MediaPipe Face Mesh Solution.
    implementation 'com.google.mediapipe:facemesh:latest.release'
    // Optional: MediaPipe Hands Solution.
    implementation 'com.google.mediapipe:hands:latest.release'
    // MediaPipe deps
    implementation 'com.google.flogger:flogger:0.6'
    implementation 'com.google.flogger:flogger-system-backend:0.6'
    implementation 'com.google.guava:guava:27.0.1-android'
    implementation 'com.google.protobuf:protobuf-java:3.11.4'
    // CameraX core library
    def camerax_version = "1.0.0-beta10"
    implementation "androidx.camera:camera-core:$camerax_version"
    implementation "androidx.camera:camera-camera2:$camerax_version"
    implementation "androidx.camera:camera-lifecycle:$camerax_version"
}
```

If you need further customization, instead of using the prebuilt maven packages
consider building a MediaPipe Android Archive library locally from source by
following these [instructions](./android_archive_library.md).

## Building solution example apps

Detailed usage examples of the Android Solution APIs can be found in the
[source code](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/solutions)
of the solution example apps.

To build these apps:

1.  Open Android Studio Arctic Fox on Linux, macOS, or Windows.

2.  Import mediapipe/examples/android/solutions directory into Android Studio.

    ![Screenshot](../images/import_mp_android_studio_project.png)

3.  For Windows users, run `create_win_symlinks.bat` as administrator to create
    res directory symlinks.

    ![Screenshot](../images/run_create_win_symlinks.png)

4.  Select "File" -> "Sync Project with Gradle Files" to sync project.

5.  Run solution example app in Android Studio.

    ![Screenshot](../images/run_android_solution_app.png)

6.  (Optional) Run solutions on CPU.

    MediaPipe solution example apps run the pipeline and model inference on GPU
    by default. If needed, for example to run the apps on Android Emulator, set
    the `RUN_ON_GPU` boolean variable to `false` in the app's
    `MainActivity.java` to run the pipeline and model inference on CPU.
