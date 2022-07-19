load("//mediapipe/render/module/render_queue:ola_aar.bzl", "ola_aar")


android_library(
    name = "RenderJni",
    srcs = ["RenderJni.java"],
    deps = [],
)

ola_aar(
    name =  "ola_render",
     srcs = [
        "RenderJni.java"
    ], 
    proguard_specs = ["proguard.pgcfg"],
    # deps = [
    #     "//mediapipe/render/module/render_queue:olarender",
    # ],
   )
