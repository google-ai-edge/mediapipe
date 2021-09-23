---
layout: default
title: Android Solutions
parent: MediaPipe on Android
grand_parent: Getting Started
nav_order: 2
---

# Android Solution APIs
{: .no_toc }

1. TOC
{:toc}
---

Please follow instructions below to use the MediaPipe Solution APIs in Android
Studio projects and build the Android example apps in the supported MediaPipe
[solutions](../solutions/solutions.md).

## Integrate MediaPipe Android Solutions in Android Studio

MediaPipe Android Solution APIs (currently in alpha) are now available in
[Google's Maven Repository](https://maven.google.com/web/index.html?#com.google.mediapipe).
To incorporate MediaPipe Android Solutions into an Android Studio project, add
the following into the project's Gradle dependencies:

```
dependencies {
    // MediaPipe solution-core is the foundation of any MediaPipe solutions.
    implementation 'com.google.mediapipe:solution-core:latest.release'
    // Optional: MediaPipe Hands solution.
    implementation 'com.google.mediapipe:hands:latest.release'
    // Optional: MediaPipe FaceMesh solution.
    implementation 'com.google.mediapipe:facemesh:latest.release'
    // MediaPipe deps
    implementation 'com.google.flogger:flogger:latest.release'
    implementation 'com.google.flogger:flogger-system-backend:latest.release'
    implementation 'com.google.guava:guava:27.0.1-android'
    implementation 'com.google.protobuf:protobuf-java:3.11.4'
    // CameraX core library
    def camerax_version = "1.0.0-beta10"
    implementation "androidx.camera:camera-core:$camerax_version"
    implementation "androidx.camera:camera-camera2:$camerax_version"
    implementation "androidx.camera:camera-lifecycle:$camerax_version"
}
```

See the detailed solutions API usage examples for different use cases in the
solution example apps'
[source code](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/solutions).
If the prebuilt maven packages are not sufficient, building the MediaPipe
Android archive library locally by following these
[instructions](./android_archive_library.md).

## Build solution example apps in Android Studio

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

    MediaPipe solution example apps run the pipeline and the model inference on
    GPU by default. If needed, for example to run the apps on Android Emulator,
    set the `RUN_ON_GPU` boolean variable to `false` in the app's
    MainActivity.java to run the pipeline and the model inference on CPU.
