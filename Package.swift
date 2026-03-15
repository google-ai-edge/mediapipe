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
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.32/MediaPipeTasksCommon.xcframework.zip",
            checksum: "84bf6892da14f25e9719ec515f46ab670ba006a4d4a039ca966352ce75b31180"
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
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.32/MediaPipeTasksVision.xcframework.zip",
            checksum: "ff3700fa7605be1e41c084bc596df3cebbec78994fd4a75fc6c478c42e9dccab"
        ),

        // MediaPipeTasksText - Text task APIs
        // Includes: text classification, text embedding, etc.
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.32/MediaPipeTasksText.xcframework.zip",
            checksum: "bdc01a87423d313fb459bbcc72d2653ac2edce41a0b5fa18cee5585ddbec7d80"
        ),

        // MediaPipeTasksAudio - Audio task APIs
        // Includes: audio classification, etc.
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.32/MediaPipeTasksAudio.xcframework.zip",
            checksum: "e3bc269be8f598dd277bee8d1647baced4896fa776ff634ce93d0c62a3027f88"
        ),
    ]
)
