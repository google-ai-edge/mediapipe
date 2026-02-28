// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MediaPipe",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        // Main products - these are what users will import
        .library(
            name: "MediaPipeTasksCommon",
            targets: ["MediaPipeTasksCommon", "MediaPipeTasksCommonWrapper"]),
        .library(
            name: "MediaPipeTasksVision",
            targets: ["MediaPipeTasksVision"]),
        .library(
            name: "MediaPipeTasksText",
            targets: ["MediaPipeTasksText"]),
        .library(
            name: "MediaPipeTasksAudio",
            targets: ["MediaPipeTasksAudio"]),
    ],
    targets: [
        // MediaPipeTasksCommon - Base framework
        // Contains core functionality shared across all task types
        .binaryTarget(
            name: "MediaPipeTasksCommon",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.26/MediaPipeTasksCommon.xcframework.zip",
            checksum: "87f2d3f00bb548d157848ddf123e4738801e2b5f3eca9dbcbb6c54b2b4c76bb3"
        ),

        // Wrapper target for MediaPipeTasksCommon to add system framework dependencies
        .target(
            name: "MediaPipeTasksCommonWrapper",
            dependencies: ["MediaPipeTasksCommon"],
            path: "Sources/MediaPipeTasksCommonWrapper",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("AssetsLibrary"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreImage"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreVideo"),
                .linkedLibrary("c++")
            ]
        ),

        // MediaPipeTasksVision - Vision task APIs
        // Includes: object detection, image classification, face detection, etc.
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.26/MediaPipeTasksVision.xcframework.zip",
            checksum: "4837cfcf14f707ceba3a8740b817c1bafd7ca874157f149ea95b8247d1b62145"
        ),

        // MediaPipeTasksText - Text task APIs
        // Includes: text classification, text embedding, etc.
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.26/MediaPipeTasksText.xcframework.zip",
            checksum: "85756cf7e27900bda822d9163a7da9d2d3c1ab53d1eeff54528184d148d5f008"
        ),

        // MediaPipeTasksAudio - Audio task APIs
        // Includes: audio classification, etc.
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.26/MediaPipeTasksAudio.xcframework.zip",
            checksum: "4da55903d264791bb57c1ddc8a0433dc3d6740d113cac5de72cb47b4a755ec6b"
        ),
    ]
)
