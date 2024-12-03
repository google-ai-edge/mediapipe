DEFAULT_VISIBILITY = [
    "//mediapipe/tasks:__subpackages__",
]

package(default_visibility = DEFAULT_VISIBILITY)

cc_library(
    name = "mdspan",
    hdrs = ["mdspan.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
