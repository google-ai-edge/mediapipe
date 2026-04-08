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
        .library(
            name: "MediaPipeTasksGenAI",
            targets: ["MediaPipeTasksGenAI"]),
        .library(
            name: "MediaPipeTasksGenAIC",
            targets: ["MediaPipeTasksGenAIC", "MediaPipeTasksGenAICWrapper"]),
    ],
    targets: [
        // MediaPipeTasksCommon - Base framework
        // Contains core functionality shared across all task types
        .binaryTarget(
            name: "MediaPipeTasksCommon",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.33/MediaPipeTasksCommon.xcframework.zip",
            checksum: "83726a0e95f354b75e8b705b8cfe85f1f169175249e117678701f9d39a642193"
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
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.33/MediaPipeTasksVision.xcframework.zip",
            checksum: "485816db112c47fd453168010601defd0dfa34375d6c2ca8a58762fb2a44ed15"
        ),

        // MediaPipeTasksText - Text task APIs
        // Includes: text classification, text embedding, etc.
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.33/MediaPipeTasksText.xcframework.zip",
            checksum: "223d03d5cdccf0f1b29502145da8f4385f74d360ddd9a4c31428569a2df7c59d"
        ),

        // MediaPipeTasksAudio - Audio task APIs
        // Includes: audio classification, etc.
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.33/MediaPipeTasksAudio.xcframework.zip",
            checksum: "74b718a05d75cb49a08fcf127ed923dda837ccac8726ae64feae141b8772c2e8"
        ),

        // MediaPipeTasksGenAI - Generative AI APIs (prebuilt, source not open)
        // Includes: LLM inference (deprecated in favor of LiteRT-LM)
        .binaryTarget(
            name: "MediaPipeTasksGenAI",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.33/MediaPipeTasksGenAI.xcframework.zip",
            checksum: "b6b1788204ccb7d24b534b09c780a6263c3f0d8b47f19bfc57b2e74f77db2de5"
        ),

        // MediaPipeTasksGenAIC - Generative AI C API (prebuilt, source not open)
        .binaryTarget(
            name: "MediaPipeTasksGenAIC",
            url: "https://github.com/mihaidimoiu/mediapipe/releases/download/v0.10.33/MediaPipeTasksGenAIC.xcframework.zip",
            checksum: "79f06cf250fe3df94d7e7a3c6e32f537fb756b1ac00a7f14848829cef2c74cbe"
        ),

        // Wrapper target for MediaPipeTasksGenAIC to add system framework and force_load dependencies
        .target(
            name: "MediaPipeTasksGenAICWrapper",
            dependencies: ["MediaPipeTasksGenAIC"],
            path: "Sources/MediaPipeTasksGenAICWrapper",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Metal"),
                .linkedFramework("OpenGLES"),
                .linkedLibrary("c++")
            ]
        ),
    ]
)
