load("//mediapipe/render/module/render_queue:ola_aar.bzl", "ola_aar")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libola_render_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/render/module/render_queue:olarender",
    ],
)

cc_library(
    name = "ola_render_jni_lib",
    srcs = [":libola_render_jni.so"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

# android_library(
#     name = "RenderJni",
#     srcs = ["RenderJni.java"],
#     deps = [],
# )




ola_aar(
    name = "ola_render",
    srcs = glob(["*.java"]),
    gen_lib = False,
    proguard_specs = ["proguard.pgcfg"],
    # deps = [
    #     "//mediapipe/render/module/render_queue:olarender",
    # ],
   )
