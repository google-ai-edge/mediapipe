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
#include "mediapipe/util/frame_buffer/halide/common.h"

namespace {

using ::mediapipe::frame_buffer::halide::common::rotate;

class RgbRotate : public Halide::Generator<RgbRotate> {
 public:
  Var x{"x"}, y{"y"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 3>> src_rgb{"src_rgb"};
  // Rotation angle in degrees counter-clockwise. Must be in {0, 90, 180, 270}.
  Input<int> rotation_angle{"rotation_angle", 0};

  Output<Func> dst_rgb{"dst_rgb", UInt(8), 3};

  void generate();
  void schedule();
};

void RgbRotate::generate() {
  const Halide::Expr width = src_rgb.dim(0).extent();
  const Halide::Expr height = src_rgb.dim(1).extent();

  // Rotate each of the RGB planes independently.
  rotate(src_rgb, dst_rgb, width, height, rotation_angle);
}

void RgbRotate::schedule() {
  // TODO: Remove specialization for (angle == 0) since that is
  // a no-op and callers should simply skip rotation. Doing so would cause
  // a bounds assertion crash if called with angle=0, however.
  Halide::Func dst_rgb_func = dst_rgb;
  Halide::Var c = dst_rgb_func.args()[2];
  Halide::OutputImageParam rgb_output = dst_rgb_func.output_buffer();
  dst_rgb_func.specialize(rotation_angle == 0).reorder(c, x, y);
  dst_rgb_func.specialize(rotation_angle == 90).reorder(c, y, x);
  dst_rgb_func.specialize(rotation_angle == 180).reorder(c, x, y);
  dst_rgb_func.specialize(rotation_angle == 270).reorder(c, y, x);

  // RGB planes starts at index zero in every dimension.
  src_rgb.dim(0).set_min(0);
  src_rgb.dim(1).set_min(0);
  src_rgb.dim(2).set_min(0);
  rgb_output.dim(0).set_min(0);
  rgb_output.dim(1).set_min(0);
  rgb_output.dim(2).set_min(0);

  // Require that the input/output buffer be interleaved and tightly-
  // packed; that is, either RGBRGBRGB[...] or RGBARGBARGBA[...],
  // without gaps between pixels.
  src_rgb.dim(0).set_stride(src_rgb.dim(2).extent());
  src_rgb.dim(2).set_stride(1);
  rgb_output.dim(0).set_stride(rgb_output.dim(2).extent());
  rgb_output.dim(2).set_stride(1);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(RgbRotate, rgb_rotate_generator)
