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

#include <cstdint>

#include "Halide.h"

namespace {

// Convert rgb_buffer between 3 and 4 channels. When converting from 3 channels
// to 4 channels, the alpha value is always 255.
class RgbRgb : public Halide::Generator<RgbRgb> {
 public:
  Var x{"x"}, y{"y"}, c{"c"};

  Input<Buffer<uint8_t, 3>> src_rgb{"src_rgb"};
  Output<Buffer<uint8_t, 3>> dst_rgb{"dst_rgb"};

  void generate();
  void schedule();
};

void RgbRgb::generate() {
  // We use Halide::clamp to avoid evaluating src_rgb(x, y, c) when c == 3 and
  // the src_rgb only has c <= 2 (rgb -> rgba conversion case).
  dst_rgb(x, y, c) =
      Halide::select(c == 3, 255, src_rgb(x, y, Halide::clamp(c, 0, 2)));
}

void RgbRgb::schedule() {
  Halide::Expr input_rgb_channels = src_rgb.dim(2).extent();
  Halide::Expr output_rgb_channels = dst_rgb.dim(2).extent();

  // The source buffer starts at zero in every dimension and requires an
  // interleaved format.
  src_rgb.dim(0).set_min(0);
  src_rgb.dim(1).set_min(0);
  src_rgb.dim(2).set_min(0);
  src_rgb.dim(0).set_stride(input_rgb_channels);
  src_rgb.dim(2).set_stride(1);

  // The destination buffer starts at zero in every dimension and requires an
  // interleaved format.
  dst_rgb.dim(0).set_min(0);
  dst_rgb.dim(1).set_min(0);
  dst_rgb.dim(2).set_min(0);
  dst_rgb.dim(0).set_stride(output_rgb_channels);
  dst_rgb.dim(2).set_stride(1);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(RgbRgb, rgb_rgb_generator)
