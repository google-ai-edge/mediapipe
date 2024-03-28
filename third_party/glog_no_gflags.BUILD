load(":bazel/glog.bzl", "glog_library")

licenses(["notice"])

# gflags is not needed on mobile platforms, and tried to link in
# -lpthread, which breaks Android builds.
# TODO: upstream.
glog_library(with_gflags = 0)
