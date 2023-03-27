# Placeholder for internal Python strict binary compatibility macro.

py_binary(
    name = "build_py_api_docs",
    srcs = ["build_py_api_docs.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/mediapipe",
        "//third_party/py/tensorflow_docs/api_generator:generate_lib",
    ],
)

py_binary(
    name = "build_java_api_docs",
    srcs = ["build_java_api_docs.py"],
    data = [
        "//third_party/android/sdk:api/26.txt",
        "//third_party/java/doclava:doclet.jar",
        "//third_party/java/jsilver:jsilver_jar",
    ],
    env = {
        "DOCLAVA_JAR": "$(location //third_party/java/doclava:doclet.jar)",
        "JSILVER_JAR": "$(location //third_party/java/jsilver:jsilver_jar)",
    },
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/tensorflow_docs/api_generator/gen_java",
    ],
)
