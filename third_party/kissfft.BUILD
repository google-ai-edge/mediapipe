# Copyright 2026 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["COPYING"])

# Float build, used by @litert_lm's audio preprocessor.
cc_library(
    name = "kissfft",
    srcs = ["kiss_fft.c"],
    hdrs = [
        "_kiss_fft_guts.h",
        "kiss_fft.h",
        "kiss_fft_log.h",
    ],
)

cc_library(
    name = "kissfftr",
    srcs = [
        "kfc.c",
        "kiss_fft.c",
        "kiss_fftnd.c",
        "kiss_fftndr.c",
        "kiss_fftr.c",
    ],
    hdrs = [
        "_kiss_fft_guts.h",
        "kfc.h",
        "kiss_fft.h",
        "kiss_fft_log.h",
        "kiss_fftnd.h",
        "kiss_fftndr.h",
        "kiss_fftr.h",
    ],
)

# Fixed point build. This repository overrides the @kissfft repository defined
# by @org_tensorflow, so it has to keep providing the target that TF Lite's
# microfrontend depends on.
cc_library(
    name = "kiss_fftr_16",
    srcs = [
        "kfc.c",
        "kiss_fft.c",
        "kiss_fftnd.c",
        "kiss_fftndr.c",
        "kiss_fftr.c",
    ],
    hdrs = [
        "_kiss_fft_guts.h",
        "kfc.h",
        "kiss_fft.h",
        "kiss_fft_log.h",
        "kiss_fftnd.h",
        "kiss_fftndr.h",
        "kiss_fftr.h",
    ],
    copts = [
        "-DFIXED_POINT=16",
    ],
)
