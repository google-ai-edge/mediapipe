package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

alias(
    name = "python_headers",
    # Point to the REAL python_headers target that rules_python provides.
    actual = "@local_config_python//:python_headers",
    visibility = ["//visibility:public"],
)
