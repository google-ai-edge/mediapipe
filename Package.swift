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
            url: "https://dl.google.com/cpdc/20261013-213039/MediaPipeTasksCommon-1.1.1.xcframework.zip",
            checksum: "25690f3005c0405a7f27b6d273b4db241811d00fbb01c54004057ee1d0e983d8"
        ),
        .binaryTarget(
            name: "MediaPipeTaskGraphsBinary",
            url: "https://dl.google.com/cpdc/20261013-213039/MediaPipeTaskGraphs-1.1.1.xcframework.zip",
            checksum: "e4d88ed65c101764ef90281a411b9942f369e48673d448eff1f934e1f68d9633"
        ),
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://dl.google.com/cpdc/20261013-213039/MediaPipeTasksVision-1.1.1.xcframework.zip",
            checksum: "755390ce853b062536899f0729321156d4cdceb263aa92c4bf3b2dd05fe640e0"
        ),
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://dl.google.com/cpdc/20261013-213039/MediaPipeTasksText-1.1.1.xcframework.zip",
            checksum: "0196f1dc59726d62ff5783ac97c42dd1f8fc07d1c94ebe72de438043d272ae9b"
        ),
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://dl.google.com/cpdc/20261013-213039/MediaPipeTasksAudio-1.1.1.xcframework.zip",
            checksum: "c163c4c6e6816f0b3ca6c4540be260e01be66e901ba65e1a029649612ce99869"
        ),
        .binaryTarget(
            name: "MediaPipeTasksRetrieval",
            url: "https://dl.google.com/cpdc/20261013-213039/MediaPipeTasksRetrieval-1.1.1.xcframework.zip",
            checksum: "fdd9e315144232a1efbaafd5394dc23b406647335b5a7c8783c6573a31f6d8f7"
        ),
        .binaryTarget(
            name: "MediaPipeTasksDecision",
            url: "https://dl.google.com/cpdc/20261013-213039/MediaPipeTasksDecision-1.1.1.xcframework.zip",
            checksum: "56b7b8e9e763bfc20dff8b4721653855d4ea9236a5f704332d0aafaa29dfc8db"
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
