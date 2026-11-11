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
            url: "https://dl.google.com/cpdc/20261102-135238/MediaPipeTasksCommon-1.2.2.xcframework.zip",
            checksum: "3ed9b3dab1008549c60cfe994a7601a3aa5f45e84dc4542c7996d840230bb47a"
        ),
        .binaryTarget(
            name: "MediaPipeTaskGraphsBinary",
            url: "https://dl.google.com/cpdc/20261102-135238/MediaPipeTaskGraphs-1.2.2.xcframework.zip",
            checksum: "eaf541888b07d5e748d93c9d1d76ee3d2418281b0bd38973d1818fb2805c4723"
        ),
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://dl.google.com/cpdc/20261102-135238/MediaPipeTasksVision-1.2.2.xcframework.zip",
            checksum: "6432a8780c5147211181039b3f9e4e1925b405e0460d43665a5510d0a941b554"
        ),
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://dl.google.com/cpdc/20261102-135238/MediaPipeTasksText-1.2.2.xcframework.zip",
            checksum: "9a05a58ab2e4674c60747ef420eee4f71bc4bb2582d7dbfc765acc1ca5da8923"
        ),
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://dl.google.com/cpdc/20261102-135238/MediaPipeTasksAudio-1.2.2.xcframework.zip",
            checksum: "e778f494c79dfe7d7dd2668d4405a2a29d4fae38ed88c29f24b3ae8ea916f603"
        ),
        .binaryTarget(
            name: "MediaPipeTasksRetrieval",
            url: "https://dl.google.com/cpdc/20261102-135238/MediaPipeTasksRetrieval-1.2.2.xcframework.zip",
            checksum: "27d7a20be92b58c5214aec2198f3f334de2504667e003f9fdde2bb5988cbe3f1"
        ),
        .binaryTarget(
            name: "MediaPipeTasksDecision",
            url: "https://dl.google.com/cpdc/20261102-135238/MediaPipeTasksDecision-1.2.2.xcframework.zip",
            checksum: "b633dfd32ed160c55373c886c671d9bf0964b00c3421d900c6f9d2a86e3723f5"
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
