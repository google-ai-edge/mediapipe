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
        .library(
            name: "MediaPipeTasksRetrieval",
            targets: [
                "MediaPipeTasksRetrieval",
                "MediaPipeTasksCommon",
            ]
        ),
        .library(
            name: "MediaPipeTasksDecision",
            targets: [
                "MediaPipeTasksDecision",
                "MediaPipeTasksCommon",
            ]
        ),
    ],
    dependencies: [],
    targets: [
        .binaryTarget(
            name: "MediaPipeTasksCommonBinary",
            url: "https://dl.google.com/cpdc/20261005-190020/MediaPipeTasksCommon-1.1.0.xcframework.zip",
            checksum: "d2194b929b91f0c866b2f8d65003e08c474f6c85e8ca7293511e9ff4a780aa68"
        ),
        .binaryTarget(
            name: "MediaPipeTaskGraphsBinary",
            url: "https://dl.google.com/cpdc/20261005-190020/MediaPipeTaskGraphs-1.1.0.xcframework.zip",
            checksum: "6e59def0a86dcf8b357d6dc4220c8f45a3704449e7c7734677f2db50671c7def"
        ),
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://dl.google.com/cpdc/20261005-190020/MediaPipeTasksVision-1.1.0.xcframework.zip",
            checksum: "d66d9929a28febdd52aba49b3336e1550e4526ab0b772c49321db05af5c7614e"
        ),
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://dl.google.com/cpdc/20261005-190020/MediaPipeTasksText-1.1.0.xcframework.zip",
            checksum: "b99e9f993a4e5c366122a62f2a14a654fa5625d6f96035cd2c1bafd4c69eaa0f"
        ),
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://dl.google.com/cpdc/20261005-190020/MediaPipeTasksAudio-1.1.0.xcframework.zip",
            checksum: "9c4506f23169a0387a745d67f4fe8872e0c381f4b67e33babd75b2187c463d11"
        ),
        .binaryTarget(
            name: "MediaPipeTasksRetrieval",
            url: "https://dl.google.com/cpdc/20261005-190020/MediaPipeTasksRetrieval-1.1.0.xcframework.zip",
            checksum: "e9703c40ce7becef0e1302a80a7fc8e6bb3205de41f367934d50f5e06f43de44"
        ),
        .binaryTarget(
            name: "MediaPipeTasksDecision",
            url: "https://dl.google.com/cpdc/20261005-190020/MediaPipeTasksDecision-1.1.0.xcframework.zip",
            checksum: "aec1c72c583d25161b68750e19656c2b4092feded7ca4acb62c76de0cc479efb"
        ),
        .target(
            name: "MediaPipeTasksCommon",
            dependencies: [
                "MediaPipeTasksCommonBinary",
                "MediaPipeTaskGraphsBinary",
                "MediaPipeTasksVision",
                "MediaPipeTasksText",
                "MediaPipeTasksAudio",
                "MediaPipeTasksRetrieval",
                "MediaPipeTasksDecision",
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
                // Required by MPPSqliteVectorStore in MediaPipeTasksRetrieval.
                .linkedLibrary("sqlite3"),
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
