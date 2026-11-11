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
            url: "https://dl.google.com/cpdc/20261111-125016/MediaPipeTasksCommon-1.2.3.xcframework.zip",
            checksum: "39990d7ff26dedeef5e0f43e7a8656ead9219b52133ae390af5e9ffcfa208b1f"
        ),
        .binaryTarget(
            name: "MediaPipeTaskGraphsBinary",
            url: "https://dl.google.com/cpdc/20261111-125016/MediaPipeTaskGraphs-1.2.3.xcframework.zip",
            checksum: "706d2c894767e281ac5c02c27f1f4bba3c1f88b25a1a0ef51d134685ae30e05c"
        ),
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://dl.google.com/cpdc/20261111-125016/MediaPipeTasksVision-1.2.3.xcframework.zip",
            checksum: "5b420e4ce53681dbccc1b7ddcda3bd90a84d3e8c5f5b465abf02968d5155ceb1"
        ),
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://dl.google.com/cpdc/20261111-125016/MediaPipeTasksText-1.2.3.xcframework.zip",
            checksum: "b7c49f1f8b563d2ddb0fd177d3dc6824f2a4703dc71534bf0db48bec62a691a0"
        ),
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://dl.google.com/cpdc/20261111-125016/MediaPipeTasksAudio-1.2.3.xcframework.zip",
            checksum: "24a7f786093ad3ed0c367ce1767e9357a0fb232168817247d4571c322548e2f9"
        ),
        .binaryTarget(
            name: "MediaPipeTasksRetrieval",
            url: "https://dl.google.com/cpdc/20261111-125016/MediaPipeTasksRetrieval-1.2.3.xcframework.zip",
            checksum: "64116abf9b5e03305118b7cb9f575823b6f31752ec6cd98d409bfab6d936f4b8"
        ),
        .binaryTarget(
            name: "MediaPipeTasksDecision",
            url: "https://dl.google.com/cpdc/20261111-125016/MediaPipeTasksDecision-1.2.3.xcframework.zip",
            checksum: "c451d11e1a49d8ca1278b00ca6bcf5e0b94daa6b07aea5a5b37a6b7de85bb186"
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
            resources: [
                .copy("PrivacyInfo.xcprivacy"),
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
