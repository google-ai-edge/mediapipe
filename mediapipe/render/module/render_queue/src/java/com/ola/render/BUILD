load("//mediapipe/render/module/render_queue:ola_aar.bzl", "ola_aar")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libola_render_jni.so",
    linkshared = 1,
    linkstatic = 1,
    srcs = [
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
    name =  "render",
    #  srcs = [
    #     "RenderJni.java"
    # ], 
    srcs = glob(["*.java"]),
    proguard_specs = ["proguard.pgcfg"],
    # deps = [
    #     "//mediapipe/render/module/render_queue:olarender",
    # ],
   )
