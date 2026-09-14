// swift-tools-version: 5.8

import PackageDescription

let package = Package(
    name: "MediaPipeTasks",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "MediaPipeTasksCommon",
            targets: ["MediaPipeTasksCommon"]
        ),
        .library(
            name: "MediaPipeTasksVision",
            targets: [
                "MediaPipeTasksVision",
                "MediaPipeTasksCommon",
            ]
        ),
        .library(
            name: "MediaPipeTasksText",
            targets: [
                "MediaPipeTasksText",
                "MediaPipeTasksCommon",
            ]
        ),
        .library(
            name: "MediaPipeTasksAudio",
            targets: [
                "MediaPipeTasksAudio",
                "MediaPipeTasksCommon",
            ]
        ),
    ],
    dependencies: [],
    targets: [
        .binaryTarget(
            name: "MediaPipeTasksCommonBinary",
            url: "https://dl.google.com/cpdc/20260911-163655/MediaPipeTasksCommon-1.0.1.xcframework.zip",
            checksum: "5c4a6a9f4c866e8456178f0707110a05484caa50e69c1d4c4e0d76d296f08e13"
        ),
        .binaryTarget(
            name: "MediaPipeTaskGraphsBinary",
            url: "https://dl.google.com/cpdc/20260911-163655/MediaPipeTaskGraphs-1.0.1.xcframework.zip",
            checksum: "673e8f5be771dd54374e90224e1ac3a0a0ac0bbb686201595d9b6c49cb21378d"
        ),
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://dl.google.com/cpdc/20260911-163655/MediaPipeTasksVision-1.0.1.xcframework.zip",
            checksum: "3ea09537d103c97ac4d40daf0b672e374ca7894fbba7fada037a8975936b9a1d"
        ),
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://dl.google.com/cpdc/20260911-163655/MediaPipeTasksText-1.0.1.xcframework.zip",
            checksum: "e9116fc78f43edd606616cd5ad983f1d9f2f6ee69df22829198a2ef1e984da8d"
        ),
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://dl.google.com/cpdc/20260911-163655/MediaPipeTasksAudio-1.0.1.xcframework.zip",
            checksum: "d458eb5bf2f84281550f0b29b0455f34851d8db2522b91926143c1b625412306"
        ),
        .target(
            name: "MediaPipeTasksCommon",
            dependencies: [
                "MediaPipeTasksCommonBinary",
                "MediaPipeTaskGraphsBinary",
                "MediaPipeTasksVision",
                "MediaPipeTasksText",
                "MediaPipeTasksAudio",
            ],
            linkerSettings: [
                // Note: -all_load is required to prevent the Apple linker
                // from stripping C++ static calculator registration
                // constructors (REGISTER_CALCULATOR) in
                // MediaPipeTaskGraphsBinary. Downstream libraries wrapping
                // MediaPipeTasks in a remote SPM package must reference this
                // package by branch/revision or local path due to Apple's
                // .unsafeFlags restriction.
                .unsafeFlags(["-Xlinker", "-all_load"]),
                .linkedLibrary("c++"),
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreImage"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("QuartzCore"),
            ]
        ),
    ]
)
