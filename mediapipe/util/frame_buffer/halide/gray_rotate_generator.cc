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

class GrayRotate : public Halide::Generator<GrayRotate> {
 public:
  Var x{"x"}, y{"y"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 2>> src_y{"src_y"};

  // Rotation angle in degrees counter-clockwise. Must be in {0, 90, 180, 270}.
  Input<int> rotation_angle{"rotation_angle", 0};

  Output<Func> dst_y{"dst_y", UInt(8), 2};

  void generate();
  void schedule();
};

void GrayRotate::generate() {
  const Halide::Expr width = src_y.dim(0).extent();
  const Halide::Expr height = src_y.dim(1).extent();

  rotate(src_y, dst_y, width, height, rotation_angle);
}

void GrayRotate::schedule() {
  Halide::Func dst_y_func = dst_y;
  dst_y_func.specialize(rotation_angle == 0).reorder(x, y);
  dst_y_func.specialize(rotation_angle == 90).reorder(y, x);
  dst_y_func.specialize(rotation_angle == 180).reorder(x, y);
  dst_y_func.specialize(rotation_angle == 270).reorder(y, x);

  // Y plane dimensions start at zero. We could additionally constrain the
  // extent to be even, but that doesn't seem to have any benefit.
  Halide::OutputImageParam dst_y_output = dst_y_func.output_buffer();
  src_y.dim(0).set_min(0);
  src_y.dim(1).set_min(0);
  dst_y_output.dim(0).set_min(0);
  dst_y_output.dim(1).set_min(0);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(GrayRotate, gray_rotate_generator)
