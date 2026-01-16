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

class GrayFlip : public Halide::Generator<GrayFlip> {
 public:
  Var x{"x"}, y{"y"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 2>> src_y{"src_y"};

  // Flip vertically if true; flips horizontally (mirroring) otherwise.
  Input<bool> flip_vertical{"flip_vertical", false};

  Output<Func> dst_y{"dst_y", UInt(8), 2};

  void generate();
  void schedule();

 private:
  void flip(Func input, Func result, Expr width, Expr height, Expr vertical);
};

void GrayFlip::generate() {
  Halide::Func flip_x, flip_y;
  flip_x(x, y, _) = src_y(src_y.dim(0).extent() - x - 1, y, _);
  flip_y(x, y, _) = src_y(x, src_y.dim(1).extent() - y - 1, _);

  dst_y(x, y, _) = select(flip_vertical, flip_y(x, y, _), flip_x(x, y, _));
}

void GrayFlip::schedule() {
  Halide::Func dst_y_func = dst_y;

  // Y plane dimensions start at zero and destination bounds must match.
  Halide::OutputImageParam dst_y_output = dst_y_func.output_buffer();
  src_y.dim(0).set_min(0);
  src_y.dim(1).set_min(0);
  dst_y_output.dim(0).set_bounds(0, src_y.dim(0).extent());
  dst_y_output.dim(1).set_bounds(0, src_y.dim(1).extent());
}

}  // namespace

HALIDE_REGISTER_GENERATOR(GrayFlip, gray_flip_generator)
