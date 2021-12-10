# Copyright 2019-2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates MediaPipe AAR including different variants of .so in jni folder.

Usage:

Creates a new mediapipe_aar() target in a BUILD file. For example,
putting the following code into mediapipe/examples/android/aar_demo/BUILD.

```
load("//mediapipe/java/com/google/mediapipe:mediapipe_aar.bzl", "mediapipe_aar")

mediapipe_aar(
    name = "demo",
    calculators = ["//mediapipe/calculators/core:pass_through_calculator"],
)
```

Then, runs the following Bazel command to generate the aar.

```
$ bazel build --strip=always -s -c opt \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --fat_apk_cpu=arm64-v8a,armeabi-v7a \
    mediapipe/examples/android/aar_demo:demo.aar
```

Finally, imports the aar into Android Studio.

"""

load("@build_bazel_rules_android//android:rules.bzl", "android_binary", "android_library")

def mediapipe_aar(
        name,
        srcs = [],
        gen_libmediapipe = True,
        calculators = [],
        assets = [],
        assets_dir = ""):
    """Generates MediaPipe android archive library.

    Args:
      name: the name of the aar.
      srcs: the additional java source code to be added into the android library.
      gen_libmediapipe: whether to generate libmediapipe_jni.so. Default to True.
      calculators: the calculator libraries to be compiled into the jni library.
      assets: additional assets to be included into the archive.
      assets_dir: path where the assets will the packaged.
    """

    # When "--define EXCLUDE_OPENCV_SO_LIB=1" is set in the build command,
    # the OpenCV so libraries will be excluded from the AAR package to
    # save the package size.
    native.config_setting(
        name = "exclude_opencv_so_lib",
        define_values = {
            "EXCLUDE_OPENCV_SO_LIB": "1",
        },
        visibility = ["//visibility:public"],
    )

    # When "--define ENABLE_STATS_LOGGING=1" is set in the build command,
    # the solution stats logging component will be added into the AAR.
    # This flag is for internal use only.
    native.config_setting(
        name = "enable_stats_logging",
        define_values = {
            "ENABLE_STATS_LOGGING": "1",
        },
        visibility = ["//visibility:public"],
    )

    _mediapipe_jni(
        name = name + "_jni",
        gen_libmediapipe = gen_libmediapipe,
        calculators = calculators,
    )

    _mediapipe_proto(
        name = name + "_proto",
    )

    native.genrule(
        name = name + "_aar_manifest_generator",
        outs = ["AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.mediapipe">
    <uses-sdk
        android:minSdkVersion="21"
        android:targetSdkVersion="27" />
</manifest>
EOF
""",
    )

    android_library(
        name = name + "_android_lib",
        srcs = srcs + [
                   "//mediapipe/java/com/google/mediapipe/components:java_src",
                   "//mediapipe/java/com/google/mediapipe/framework:java_src",
                   "//mediapipe/java/com/google/mediapipe/glutil:java_src",
                   "com/google/mediapipe/formats/annotation/proto/RasterizationProto.java",
                   "com/google/mediapipe/formats/proto/ClassificationProto.java",
                   "com/google/mediapipe/formats/proto/DetectionProto.java",
                   "com/google/mediapipe/formats/proto/LandmarkProto.java",
                   "com/google/mediapipe/formats/proto/LocationDataProto.java",
                   "com/google/mediapipe/proto/CalculatorProto.java",
               ] +
               select({
                   "//conditions:default": [],
                   "enable_stats_logging": [
                       "com/google/mediapipe/proto/MediaPipeLoggingProto.java",
                       "com/google/mediapipe/proto/MediaPipeLoggingEnumsProto.java",
                   ],
               }),
        manifest = "AndroidManifest.xml",
        proguard_specs = ["//mediapipe/java/com/google/mediapipe/framework:proguard.pgcfg"],
        deps = [
            ":" + name + "_jni_cc_lib",
            "//mediapipe/framework:calculator_java_proto_lite",
            "//mediapipe/framework:calculator_profile_java_proto_lite",
            "//mediapipe/framework:calculator_options_java_proto_lite",
            "//mediapipe/framework:mediapipe_options_java_proto_lite",
            "//mediapipe/framework:packet_factory_java_proto_lite",
            "//mediapipe/framework:packet_generator_java_proto_lite",
            "//mediapipe/framework:status_handler_java_proto_lite",
            "//mediapipe/framework:stream_handler_java_proto_lite",
            "//mediapipe/framework/tool:calculator_graph_template_java_proto_lite",
            "//mediapipe/java/com/google/mediapipe/components:android_components",
            "//mediapipe/java/com/google/mediapipe/components:android_camerax_helper",
            "//mediapipe/java/com/google/mediapipe/framework:android_framework",
            "//mediapipe/java/com/google/mediapipe/glutil",
            "//third_party:androidx_annotation",
            "//third_party:androidx_appcompat",
            "//third_party:androidx_core",
            "//third_party:androidx_legacy_support_v4",
            "//third_party:autovalue",
            "//third_party:camerax_core",
            "//third_party:camerax_camera2",
            "//third_party:camerax_lifecycle",
            "@com_google_protobuf//:protobuf_javalite",
            "@maven//:com_google_code_findbugs_jsr305",
            "@maven//:com_google_flogger_flogger",
            "@maven//:com_google_flogger_flogger_system_backend",
            "@maven//:com_google_guava_guava",
            "@maven//:androidx_lifecycle_lifecycle_common",
        ] + select({
            "//conditions:default": [":" + name + "_jni_opencv_cc_lib"],
            "//mediapipe/framework/port:disable_opencv": [],
            "exclude_opencv_so_lib": [],
        }) + select({
            "//conditions:default": [],
            "enable_stats_logging": [
                "@maven//:com_google_android_datatransport_transport_api",
                "@maven//:com_google_android_datatransport_transport_backend_cct",
                "@maven//:com_google_android_datatransport_transport_runtime",
            ],
        }),
        assets = assets,
        assets_dir = assets_dir,
    )

    _aar_with_jni(name, name + "_android_lib")

def _mediapipe_proto(name):
    """Generates MediaPipe java proto libraries.

    Args:
      name: the name of the target.
    """
    _proto_java_src_generator(
        name = "mediapipe_log_extension_proto",
        proto_src = "mediapipe/util/analytics/mediapipe_log_extension.proto",
        java_lite_out = "com/google/mediapipe/proto/MediaPipeLoggingProto.java",
        srcs = ["//mediapipe/util/analytics:protos_src"],
    )

    _proto_java_src_generator(
        name = "mediapipe_logging_enums_proto",
        proto_src = "mediapipe/util/analytics/mediapipe_logging_enums.proto",
        java_lite_out = "com/google/mediapipe/proto/MediaPipeLoggingEnumsProto.java",
        srcs = ["//mediapipe/util/analytics:protos_src"],
    )

    _proto_java_src_generator(
        name = "calculator_proto",
        proto_src = "mediapipe/framework/calculator.proto",
        java_lite_out = "com/google/mediapipe/proto/CalculatorProto.java",
        srcs = ["//mediapipe/framework:protos_src"],
    )

    _proto_java_src_generator(
        name = "landmark_proto",
        proto_src = "mediapipe/framework/formats/landmark.proto",
        java_lite_out = "com/google/mediapipe/formats/proto/LandmarkProto.java",
        srcs = ["//mediapipe/framework/formats:protos_src"],
    )

    _proto_java_src_generator(
        name = "rasterization_proto",
        proto_src = "mediapipe/framework/formats/annotation/rasterization.proto",
        java_lite_out = "com/google/mediapipe/formats/annotation/proto/RasterizationProto.java",
        srcs = ["//mediapipe/framework/formats/annotation:protos_src"],
    )

    _proto_java_src_generator(
        name = "location_data_proto",
        proto_src = "mediapipe/framework/formats/location_data.proto",
        java_lite_out = "com/google/mediapipe/formats/proto/LocationDataProto.java",
        srcs = [
            "//mediapipe/framework/formats:protos_src",
            "//mediapipe/framework/formats/annotation:protos_src",
        ],
    )

    _proto_java_src_generator(
        name = "detection_proto",
        proto_src = "mediapipe/framework/formats/detection.proto",
        java_lite_out = "com/google/mediapipe/formats/proto/DetectionProto.java",
        srcs = [
            "//mediapipe/framework/formats:protos_src",
            "//mediapipe/framework/formats/annotation:protos_src",
        ],
    )

    _proto_java_src_generator(
        name = "classification_proto",
        proto_src = "mediapipe/framework/formats/classification.proto",
        java_lite_out = "com/google/mediapipe/formats/proto/ClassificationProto.java",
        srcs = [
            "//mediapipe/framework/formats:protos_src",
        ],
    )

def _proto_java_src_generator(name, proto_src, java_lite_out, srcs = []):
    native.genrule(
        name = name + "_proto_java_src_generator",
        srcs = srcs + [
            "@com_google_protobuf//:lite_well_known_protos",
        ],
        outs = [java_lite_out],
        cmd = "$(location @com_google_protobuf//:protoc) " +
              "--proto_path=. --proto_path=$(GENDIR) " +
              "--proto_path=$$(pwd)/external/com_google_protobuf/src " +
              "--java_out=lite:$(GENDIR) " + proto_src + " && " +
              "mv $(GENDIR)/" + java_lite_out + " $$(dirname $(location " + java_lite_out + "))",
        tools = [
            "@com_google_protobuf//:protoc",
        ],
    )

def _mediapipe_jni(name, gen_libmediapipe, calculators = []):
    """Generates MediaPipe jni library.

    Args:
      name: the name of the target.
      gen_libmediapipe: whether to generate libmediapipe_jni.so. Default to True.
      calculators: the calculator libraries to be compiled into the jni library.
    """
    if gen_libmediapipe:
        native.cc_binary(
            name = "libmediapipe_jni.so",
            linkshared = 1,
            linkstatic = 1,
            deps = [
                "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
            ] + calculators,
        )

    native.cc_library(
        name = name + "_cc_lib",
        srcs = [":libmediapipe_jni.so"],
        alwayslink = 1,
    )

    native.cc_library(
        name = name + "_opencv_cc_lib",
        srcs = select({
            "//mediapipe:android_arm64": ["@android_opencv//:libopencv_java3_so_arm64-v8a"],
            "//mediapipe:android_armeabi": ["@android_opencv//:libopencv_java3_so_armeabi-v7a"],
            "//mediapipe:android_arm": ["@android_opencv//:libopencv_java3_so_armeabi-v7a"],
            "//mediapipe:android_x86": ["@android_opencv//:libopencv_java3_so_x86"],
            "//mediapipe:android_x86_64": ["@android_opencv//:libopencv_java3_so_x86_64"],
            "//conditions:default": [],
        }),
        alwayslink = 1,
    )

def _aar_with_jni(name, android_library):
    # Generates dummy AndroidManifest.xml for dummy apk usage
    # (dummy apk is generated by <name>_dummy_app target below)
    native.genrule(
        name = name + "_binary_manifest_generator",
        outs = [name + "_generated_AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
  package="dummy.package.for.so">
  <uses-sdk android:minSdkVersion="21"/>
</manifest>
EOF
""",
    )

    # Generates dummy apk including .so files.
    # We extract out .so files and throw away the apk.
    android_binary(
        name = name + "_dummy_app",
        manifest = name + "_generated_AndroidManifest.xml",
        custom_package = "dummy.package.for.so",
        multidex = "native",
        deps = [android_library],
    )

    native.genrule(
        name = name,
        srcs = [android_library + ".aar", name + "_dummy_app_unsigned.apk"],
        outs = [name + ".aar"],
        tags = ["manual"],
        cmd = """
cp $(location {}.aar) $(location :{}.aar)
chmod +w $(location :{}.aar)
origdir=$$PWD
cd $$(mktemp -d)
unzip $$origdir/$(location :{}_dummy_app_unsigned.apk) "lib/*"
cp -r lib jni
zip -r $$origdir/$(location :{}.aar) jni/*/*.so
""".format(android_library, name, name, name, name),
    )
