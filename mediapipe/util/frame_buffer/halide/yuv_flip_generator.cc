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

class YuvFlip : public Halide::Generator<YuvFlip> {
 public:
  Var x{"x"}, y{"y"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 2>> src_y{"src_y"};
  Input<Buffer<uint8_t, 3>> src_uv{"src_uv"};
  // Flip vertically if true; flips horizontally (mirroring) otherwise.
  Input<bool> flip_vertical{"flip_vertical", false};

  Output<Func> dst_y{"dst_y", UInt(8), 2};
  Output<Func> dst_uv{"dst_uv", UInt(8), 3};

  void generate();
  void schedule();

 private:
  void flip(Func input, Func result, Expr width, Expr height, Expr vertical);
};

void YuvFlip::flip(Halide::Func input, Halide::Func result, Halide::Expr width,
                   Halide::Expr height, Halide::Expr vertical) {
  Halide::Func flip_x, flip_y;
  flip_x(x, y, _) = input(width - x - 1, y, _);
  flip_y(x, y, _) = input(x, height - y - 1, _);

  result(x, y, _) = select(vertical, flip_y(x, y, _), flip_x(x, y, _));
}

void YuvFlip::generate() {
  const Halide::Expr width = src_y.dim(0).extent();
  const Halide::Expr height = src_y.dim(1).extent();

  // Flip each of the YUV planes independently.
  flip(src_y, dst_y, width, height, flip_vertical);
  flip(src_uv, dst_uv, (width + 1) / 2, (height + 1) / 2, flip_vertical);
}

void YuvFlip::schedule() {
  Halide::Func dst_y_func = dst_y;
  Halide::Func dst_uv_func = dst_uv;
  Halide::Var c = dst_uv_func.args()[2];
  dst_uv_func.unroll(c);
  dst_uv_func.reorder(c, x, y);

  // Y plane dimensions start at zero and destination bounds must match.
  Halide::OutputImageParam dst_y_output = dst_y_func.output_buffer();
  src_y.dim(0).set_min(0);
  src_y.dim(1).set_min(0);
  dst_y_output.dim(0).set_bounds(0, src_y.dim(0).extent());
  dst_y_output.dim(1).set_bounds(0, src_y.dim(1).extent());

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

HALIDE_REGISTER_GENERATOR(YuvFlip, yuv_flip_generator)
