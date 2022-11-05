load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application",
    "ios_dynamic_framework",
    "ios_framework",
    "ios_static_framework",
    "ios_unit_test",
)
load(
    "//mediapipe/examples/ios:bundle_id.bzl",
    "BUNDLE_ID_PREFIX",
    "example_provisioning",
)

MPP_HEADERS = [
    "PoseTracking.h",
    "PoseTrackingOptions.h",
    "PoseTrackingResults.h",
]

MP_IOS_HEADERS_NAMES = [
    "MPPCameraInputSource.h",
    "MPPLayerRenderer.h",
    "MPPPlayerInputSource.h",
    "MPPInputSource.h",
    "MPPGLViewRenderer.h",
]

MP_IOS_HEADERS = ["//mediapipe/objc:" + header for header in MP_IOS_HEADERS_NAMES]

#[genrule(
#    name = "gen_" + header_name,
#    srcs = [header_rule],
#    outs = [header_name],
#    cmd = """
#    OUTPUT_DIR=mediapipe/objc/solutions/posetracking_gpu
#    mkdir -p $$OUTPUT_DIR
#    OUTPUT_PATH=$$OUTPUT_DIR/{}
#    cp $(location {})  $$OUTPUT_PATH
#    sed -i -e "s#mediapipe/objc/##g" $$OUTPUT_PATH
#    cp $$OUTPUT_PATH $@
#    """.format(header_name, header_rule),
#    local = False,
#) for (header_name, header_rule) in zip(MP_IOS_HEADERS_NAMES, MP_IOS_HEADERS)]
#
#MP_GEN_IOS_HEADERS = ["gen_" + header for header in MP_IOS_HEADERS_NAMES]
#
#filegroup(
#    name = "gen_mp_ios_headers",
#    srcs = MP_GEN_IOS_HEADERS,
#)

#-c opt --config=ios_fat --cxxopt=--std=c++17 --copt=-fembed-bitcode --linkopt="-s"
MP_GEN_IOS_HEADERS = MP_IOS_HEADERS

ios_static_framework(
    name = "MPPoseTracking",
    hdrs = MPP_HEADERS + MP_GEN_IOS_HEADERS,
    bundle_name = "MPPoseTracking",
    linkopts = [
        #        "--no-whole-archive",
        #        "-all_load",
        #        "-Wl",
        "-force_load",
    ],
    minimum_os_version = "12.0",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc/solutions/posetracking_gpu:posetracking_gpu_solution",
        "//mediapipe/calculators/core:flow_limiter_calculator",

        #        "//third_party:opencv",
        "@ios_opencv//:OpencvFramework",
    ],
)

ios_framework(
    name = "MPPoseTrackingDynamic",
    hdrs = MPP_HEADERS + MP_GEN_IOS_HEADERS + ["MPPoseTracking.h"],
    bundle_id = BUNDLE_ID_PREFIX + ".MPPPoseTracking",
    bundle_name = "MPPoseTracking",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    linkopts = [
        #        "--no-whole-archive",
        #        "-all_load",
        #        "-Wl",
        #        "-force_load",
    ],
    minimum_os_version = "10.0",
    resources = ["modulemap/module.modulemap"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc/solutions/posetracking_gpu:posetracking_gpu_solution",
        #        "//mediapipe/calculators/core:flow_limiter_calculator",

        #        "//third_party:opencv",
        "@ios_opencv//:OpencvFramework",
    ],
)

genrule(
    name = "PatchedMPPosetrackingDynamic",
    srcs = [":MPPoseTrackingDynamic"],
    outs = [
        "MPPoseTracking.framework",
        "MPPoseTracking.framework.dSYM",
    ],
    cmd = """
        bash $(location patch_ios_framework.sh) "$(SRCS)" "$(OUTS)"
    """,
    output_to_bindir = 1,
    tools = ["patch_ios_framework.sh"],
)

# Custom Bazel Rule that patches headers of framework to flatten header imports
genrule(
    name = "MPPoseTrackingHeaderPatched",
    srcs = [":MPPoseTracking"],
    outs = ["MPPoseTrackingHeaderPatched.zip"],
    cmd = """
    unzip $(location MPPoseTracking)
    sed -i -e "s#mediapipe/objc/##g" $$(find MPPoseTracking.framework -name "*.h")
    rm -f MPPoseTracking.framework/Headers/*.h-e
    zip -r MPPoseTrackingHeaderPatched.zip MPPoseTracking.framework
    cp MPPoseTrackingHeaderPatched.zip $@
    """,
    visibility = ["//visibility:public"],
)

objc_library(
    name = "posetracking_gpu_solution",
    srcs = glob([
        "*.h",
        "*.mm",
    ]),
    hdrs = MPP_HEADERS + MP_GEN_IOS_HEADERS,
    copts = [
        "-Wno-shorten-64-to-32",
        #        "-all_load",
    ],
    data = [
        "//mediapipe/graphs/pose_tracking:pose_tracking_gpu.binarypb",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_heavy.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_lite.tflite",
    ],
    linkopts = [
        #        "--no-whole-archive",
        #        "-all_load",
        #        "-force_load",
        #        "-Wl",
    ],
    module_name = "MPPoseTracking",
    sdk_frameworks = [
        "Accelerate",
        "AVFoundation",
        "CoreGraphics",
        "CoreMedia",
        "UIKit",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/pose_tracking:pose_tracking_gpu_deps",
            "//mediapipe/framework/formats:landmark_cc_proto",
            "calculator_registry",
        ],
    }),
    alwayslink = True,
)

cc_library(
    name = "calculator_registry",
    srcs = [
        "registry/calculator_registry.cpp",
        "registry/calculator_registry.h",
    ],
    deps = ["//mediapipe/graphs/pose_tracking:pose_tracking_gpu_deps"],
    alwayslink = True,
)

exports_files(
    MPP_HEADERS,
    visibility = ["//visibility:public"],
)
