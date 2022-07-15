cc_library(
    name = "ios_olarender",
    srcs = [
        "util.cpp",
        "image_queue.cpp",
        "image.cpp",
        "OlaRender.cpp",
        "OlaRenderIMP.cpp",
        ],
    hdrs = [
        "util.h", 
        "macros.h",
        "image_queue.h",
        "image.h",
        "LockFreeQueue.h",
        "OlaRender.hpp",
        "OlaRenderIMP.hpp",
        ],
    # Use -Dverbose=-1 to turn off zlib's trace logging. (#3280)
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/modules/render_queue/math:math",
    ],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
)

