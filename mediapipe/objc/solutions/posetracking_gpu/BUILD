load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_framework", "ios_static_framework", "ios_unit_test")

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
    ],
    minimum_os_version = "12.0",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc/solutions/posetracking_gpu:posetracking_gpu_solution",
        "//mediapipe/calculators/core:flow_limiter_calculator",

        #        "//third_party:opencv",
        #        "@ios_opencv//:OpencvFramework",
    ],
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
        #        "-Wl",
    ],
    module_name = "MPPoseTracking",
    sdk_frameworks = ["Accelerate"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
        "//mediapipe/calculators/core:flow_limiter_calculator",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/pose_tracking:pose_tracking_gpu_deps",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
    alwayslink = True,
)

exports_files(
    MPP_HEADERS,
    visibility = ["//visibility:public"],
)
