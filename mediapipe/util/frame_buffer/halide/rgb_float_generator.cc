// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Halide.h"

namespace {

class RgbFloat : public Halide::Generator<RgbFloat> {
 public:
  Var x{"x"}, y{"y"}, c{"c"};

  Input<Buffer<uint8_t, 3>> src_rgb{"src_rgb"};
  Input<float> scale{"scale"};
  Input<float> offset{"offset"};
  Output<Buffer<float, 3>> dst_float{"dst_float"};

  void generate();
  void schedule();
};

void RgbFloat::generate() {
  dst_float(x, y, c) = Halide::cast<float>(src_rgb(x, y, c)) * scale + offset;
}

void RgbFloat::schedule() {
  Halide::Expr input_rgb_channels = src_rgb.dim(2).extent();
  Halide::Expr output_float_channels = dst_float.dim(2).extent();

  // The source buffer starts at zero in every dimension and requires an
  // interleaved format.
  src_rgb.dim(0).set_min(0);
  src_rgb.dim(1).set_min(0);
  src_rgb.dim(2).set_min(0);
  src_rgb.dim(0).set_stride(input_rgb_channels);
  src_rgb.dim(2).set_stride(1);

  // The destination buffer starts at zero in every dimension and requires an
  // interleaved format.
  dst_float.dim(0).set_min(0);
  dst_float.dim(1).set_min(0);
  dst_float.dim(2).set_min(0);
  dst_float.dim(0).set_stride(output_float_channels);
  dst_float.dim(2).set_stride(1);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(RgbFloat, rgb_float_generator)
