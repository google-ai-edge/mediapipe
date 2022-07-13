
cc_library(
    name = "math",
    srcs = glob(["*.cpp","*.inl"]),
    hdrs = glob(["*.hpp"]),
    # Use -Dverbose=-1 to turn off zlib's trace logging. (#3280)
    includes = ["."],
    visibility = ["//visibility:public"],
)