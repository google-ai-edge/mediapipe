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

using ::Halide::_;

class RgbFlip : public Halide::Generator<RgbFlip> {
 public:
  Var x{"x"}, y{"y"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 3>> src_rgb{"src_rgb"};
  // Flip vertically if true; flips horizontally (mirroring) otherwise.
  Input<bool> flip_vertical{"flip_vertical", false};

  Output<Func> dst_rgb{"dst_rgb", UInt(8), 3};

  void generate();
  void schedule();

 private:
  void flip(Func input, Func result, Expr width, Expr height, Expr vertical);
};

void RgbFlip::flip(Halide::Func input, Halide::Func result, Halide::Expr width,
                   Halide::Expr height, Halide::Expr vertical) {
  Halide::Func flip_x, flip_y;
  flip_x(x, y, _) = input(width - x - 1, y, _);
  flip_y(x, y, _) = input(x, height - y - 1, _);

  result(x, y, _) = select(vertical, flip_y(x, y, _), flip_x(x, y, _));
}

void RgbFlip::generate() {
  const Halide::Expr width = src_rgb.dim(0).extent();
  const Halide::Expr height = src_rgb.dim(1).extent();

  // Flip each of the RGB planes independently.
  flip(src_rgb, dst_rgb, width, height, flip_vertical);
}

void RgbFlip::schedule() {
  Halide::Func dst_rgb_func = dst_rgb;
  Halide::Var c = dst_rgb_func.args()[2];
  Halide::OutputImageParam rgb_output = dst_rgb_func.output_buffer();

  // Iterate over channel in the innermost loop, then x, then y.
  dst_rgb_func.reorder(c, x, y);

  // RGB planes starts at index zero in every dimension and destination bounds
  // must match.
  src_rgb.dim(0).set_min(0);
  src_rgb.dim(1).set_min(0);
  src_rgb.dim(2).set_min(0);
  rgb_output.dim(0).set_bounds(0, src_rgb.dim(0).extent());
  rgb_output.dim(1).set_bounds(0, src_rgb.dim(1).extent());
  rgb_output.dim(2).set_bounds(0, src_rgb.dim(2).extent());

  // Require that the input/output buffer be interleaved and tightly-
  // packed; that is, either RGBRGBRGB[...] or RGBARGBARGBA[...],
  // without gaps between pixels.
  src_rgb.dim(0).set_stride(src_rgb.dim(2).extent());
  src_rgb.dim(2).set_stride(1);
  rgb_output.dim(0).set_stride(rgb_output.dim(2).extent());
  rgb_output.dim(2).set_stride(1);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(RgbFlip, rgb_flip_generator)
