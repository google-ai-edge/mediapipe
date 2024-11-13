cc_library(
    name = "serialization",
    hdrs = [
        "result_serialization.h"
    ],
    deps = [
        "//mediapipe/calculators/geti/utils:data_structures",
        "//third_party/cpp-base64:cpp-base64",
        "@nlohmann_json//:json",
    ]
)

filegroup(
    name = "test_data",
    srcs = glob(["*.json"]),
)

cc_library(
    name = "calculators",
    hdrs = [
        "serialization_calculators.h",
        "httpserialization_calculator.h",
    ],
    srcs = [
        "serialization_calculators.cc",
        "httpserialization_calculator.cc",
    ],
    deps = [
        "//mediapipe/calculators/geti/inference:inference_calculators",
        "//mediapipe/calculators/geti/inference:geti_calculator_base",
        "//mediapipe/calculators/geti/inference:inference_utils",
        ":serialization"
    ],
    visibility = [
        "//visibility:public",
    ],
    alwayslink=1
)

cc_test(
    name = "serialization_calculators_test",
    srcs = ["serialization_calculators_test.cc" ],
    data = [
        ":test_data"
    ],
    deps = [
        ":calculators",
        "//mediapipe/calculators/geti/inference:test_deps"
    ]
)


