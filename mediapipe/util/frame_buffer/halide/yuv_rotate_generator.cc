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

class YuvRotate : public Halide::Generator<YuvRotate> {
 public:
  Var x{"x"}, y{"y"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 2>> src_y{"src_y"};
  Input<Buffer<uint8_t, 3>> src_uv{"src_uv"};
  // Rotation angle in degrees counter-clockwise. Must be in {0, 90, 180, 270}.
  Input<int> rotation_angle{"rotation_angle", 0};

  Output<Func> dst_y{"dst_y", UInt(8), 2};
  Output<Func> dst_uv{"dst_uv", UInt(8), 3};

  void generate();
  void schedule();
};

void YuvRotate::generate() {
  const Halide::Expr width = src_y.dim(0).extent();
  const Halide::Expr height = src_y.dim(1).extent();

  // Rotate each of the YUV planes independently.
  rotate(src_y, dst_y, width, height, rotation_angle);
  rotate(src_uv, dst_uv, (width + 1) / 2, (height + 1) / 2, rotation_angle);
}

void YuvRotate::schedule() {
  // TODO: Remove specialization for (angle == 0) since that is
  // a no-op and callers should simply skip rotation. Doing so would cause
  // a bounds assertion crash if called with angle=0, however.
  Halide::Func dst_y_func = dst_y;
  dst_y_func.specialize(rotation_angle == 0).reorder(x, y);
  dst_y_func.specialize(rotation_angle == 90).reorder(y, x);
  dst_y_func.specialize(rotation_angle == 180).reorder(x, y);
  dst_y_func.specialize(rotation_angle == 270).reorder(y, x);

  Halide::Func dst_uv_func = dst_uv;
  Halide::Var c = dst_uv_func.args()[2];
  dst_uv_func.unroll(c);
  dst_uv_func.specialize(rotation_angle == 0).reorder(c, x, y);
  dst_uv_func.specialize(rotation_angle == 90).reorder(c, y, x);
  dst_uv_func.specialize(rotation_angle == 180).reorder(c, x, y);
  dst_uv_func.specialize(rotation_angle == 270).reorder(c, y, x);

  // Y plane dimensions start at zero. We could additionally constrain the
  // extent to be even, but that doesn't seem to have any benefit.
  Halide::OutputImageParam dst_y_output = dst_y_func.output_buffer();
  src_y.dim(0).set_min(0);
  src_y.dim(1).set_min(0);
  dst_y_output.dim(0).set_min(0);
  dst_y_output.dim(1).set_min(0);

  // UV plane has two channels and is half the size of the Y plane in X/Y.
  Halide::OutputImageParam dst_uv_output = dst_uv_func.output_buffer();
  src_uv.dim(0).set_bounds(0, (src_y.dim(0).extent() + 1) / 2);
  src_uv.dim(1).set_bounds(0, (src_y.dim(1).extent() + 1) / 2);
  src_uv.dim(2).set_bounds(0, 2);
  dst_uv_output.dim(0).set_bounds(0, (dst_y_output.dim(0).extent() + 1) / 2);
  dst_uv_output.dim(1).set_bounds(0, (dst_y_output.dim(1).extent() + 1) / 2);
  dst_uv_output.dim(2).set_bounds(0, 2);

  // Remove default memory layout constraints and accept/produce generic UV
  // (including semi-planar and planar).
  src_uv.dim(0).set_stride(Expr());
  dst_uv_output.dim(0).set_stride(Expr());
}

}  // namespace

HALIDE_REGISTER_GENERATOR(YuvRotate, yuv_rotate_generator)
