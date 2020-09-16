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

"""Generate MediaPipe AAR including different variants of .so in jni folder.

Usage:

Create a new mediapipe_aar() target in a BUILD file. For example,
putting the following code into mediapipe/examples/android/aar_demo/BUILD.

```
load("//mediapipe/java/com/google/mediapipe:mediapipe_aar.bzl", "mediapipe_aar")

mediapipe_aar(
    name = "my_aar",
    calculators = ["//mediapipe/calculators/core:pass_through_calculator"],
)
```

Then, run the following Bazel command to generate the AAR.

```
$ bazel build -c opt --fat_apk_cpu=arm64-v8a,armeabi-v7a mediapipe/examples/android/aar_demo:my_aar
```

Finally, import the AAR into Android Studio.

"""

load("@build_bazel_rules_android//android:rules.bzl", "android_binary", "android_library")

def mediapipe_aar(name, calculators = [], assets = [], assets_dir = ""):
    """Generate MediaPipe AAR.

    Args:
      name: the name of the AAR.
      calculators: the calculator libraries to be compiled into the .so.
      assets: additional assets to be included into the archive.
      assets_dir: path where the assets will the packaged.
    """
    native.cc_binary(
        name = "libmediapipe_jni.so",
        linkshared = 1,
        linkstatic = 1,
        deps = [
            "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        ] + calculators,
    )

    native.cc_library(
        name = name + "_mediapipe_jni_lib",
        srcs = [":libmediapipe_jni.so"],
        alwayslink = 1,
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
    <application />
</manifest>
""",
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

    android_library(
        name = name + "_android_lib",
        srcs = [
            "//mediapipe/java/com/google/mediapipe/components:java_src",
            "//mediapipe/java/com/google/mediapipe/framework:java_src",
            "//mediapipe/java/com/google/mediapipe/glutil:java_src",
            "com/google/mediapipe/proto/CalculatorProto.java",
            "com/google/mediapipe/formats/proto/LandmarkProto.java",
            "com/google/mediapipe/formats/proto/DetectionProto.java",
            "com/google/mediapipe/formats/proto/LocationDataProto.java",
            "com/google/mediapipe/formats/annotation/proto/RasterizationProto.java",
        ],
        manifest = "AndroidManifest.xml",
        proguard_specs = ["//mediapipe/java/com/google/mediapipe/framework:proguard.pgcfg"],
        deps = [
            ":" + name + "_mediapipe_jni_lib",
            "//mediapipe/framework:calculator_java_proto_lite",
            "//mediapipe/framework:calculator_profile_java_proto_lite",
            "//mediapipe/framework/tool:calculator_graph_template_java_proto_lite",
            "//third_party:androidx_annotation",
            "//third_party:androidx_appcompat",
            "//third_party:androidx_core",
            "//third_party:androidx_legacy_support_v4",
            "//third_party:camerax_core",
            "//third_party:camera2",
            "@maven//:com_google_code_findbugs_jsr305",
            "@maven//:com_google_flogger_flogger",
            "@maven//:com_google_flogger_flogger_system_backend",
            "@maven//:com_google_guava_guava",
            "@maven//:androidx_lifecycle_lifecycle_common",
        ],
        assets = assets,
        assets_dir = assets_dir,
    )

    _aar_with_jni(name, name + "_android_lib")

def _proto_java_src_generator(name, proto_src, java_lite_out, srcs = []):
    native.genrule(
        name = name + "_proto_java_src_generator",
        srcs = srcs + [
            "@com_google_protobuf//:well_known_protos",
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

def _aar_with_jni(name, android_library):
    # Generate dummy AndroidManifest.xml for dummy apk usage
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

    # Generate dummy apk including .so files.
    # We extract out .so files and throw away the apk.
    android_binary(
        name = name + "_dummy_app",
        manifest = name + "_generated_AndroidManifest.xml",
        custom_package = "dummy.package.for.so",
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
