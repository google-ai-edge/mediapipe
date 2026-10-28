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
            url: "https://dl.google.com/cpdc/20261026-192222/MediaPipeTasksCommon-1.2.0.xcframework.zip",
            checksum: "4785f1fe4d4caa1abe393f32522984674c391fccbaf11a8508f62bf1e4cc4b25"
        ),
        .binaryTarget(
            name: "MediaPipeTaskGraphsBinary",
            url: "https://dl.google.com/cpdc/20261026-192222/MediaPipeTaskGraphs-1.2.0.xcframework.zip",
            checksum: "c8a63ec2f899dd90555558c02565440bb0001842f54038dcb8dc2868447c0ca5"
        ),
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://dl.google.com/cpdc/20261026-192222/MediaPipeTasksVision-1.2.0.xcframework.zip",
            checksum: "3e621f63ced11c18745182f1f2daf256529a27d484ec8ceea542bf8bb3c6a2c4"
        ),
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://dl.google.com/cpdc/20261026-192222/MediaPipeTasksText-1.2.0.xcframework.zip",
            checksum: "b540c11451099fce15764303ed0ca88decb1494bdcbd53fd7fc7a4d48a71aaac"
        ),
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://dl.google.com/cpdc/20261026-192222/MediaPipeTasksAudio-1.2.0.xcframework.zip",
            checksum: "5590bee9c1c41b46f79c27fc17bf298f13b8a381354310ddab1439791a3e2f96"
        ),
        .binaryTarget(
            name: "MediaPipeTasksRetrieval",
            url: "https://dl.google.com/cpdc/20261026-192222/MediaPipeTasksRetrieval-1.2.0.xcframework.zip",
            checksum: "9780708496a05e284f6a9aaa6e0122f8d1d170c291139aea15ad65a22dc602e3"
        ),
        .binaryTarget(
            name: "MediaPipeTasksDecision",
            url: "https://dl.google.com/cpdc/20261026-192222/MediaPipeTasksDecision-1.2.0.xcframework.zip",
            checksum: "77abefeab03cce66bd9a017e323cc37bc1e0367a61788e4dcd5e96c7d109905c"
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
