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

#include "mediapipe/util/frame_buffer/halide/common.h"

namespace mediapipe {
namespace frame_buffer {
namespace halide {
namespace common {

namespace {
using ::Halide::_;
}

void resize_nn(Halide::Func input, Halide::Func result, Halide::Expr fx,
               Halide::Expr fy) {
  Halide::Var x{"x"}, y{"y"};
  result(x, y, _) = input(Halide::cast<int>((x + 0.5f) * fx),
                          Halide::cast<int>((y + 0.5f) * fy), _);
}

// Borrowed from photos/editing/halide/src/resize_image_bilinear_generator.cc:
void resize_bilinear(Halide::Func input, Halide::Func result, Halide::Expr fx,
                     Halide::Expr fy) {
  Halide::Var x{"x"}, y{"y"};
  Halide::Func x_interpolated("x_interpolated");

  Halide::Expr xi = Halide::cast<int>(x * fx);
  Halide::Expr xr = x * fx - xi;
  Halide::Expr x0 = input(xi + 0, y, _);
  Halide::Expr x1 = input(xi + 1, y, _);
  x_interpolated(x, y, _) = lerp(x0, x1, xr);

  Halide::Expr yi = Halide::cast<int>(y * fy);
  Halide::Expr yr = y * fy - yi;
  Halide::Expr y0 = x_interpolated(x, yi + 0, _);
  Halide::Expr y1 = x_interpolated(x, yi + 1, _);
  result(x, y, _) = lerp(y0, y1, yr);
}

void resize_bilinear_int(Halide::Func input, Halide::Func result,
                         Halide::Expr fx, Halide::Expr fy) {
  Halide::Var x{"x"}, y{"y"};
  Halide::Func x_interpolated("x_interpolated");

  fx = Halide::cast<int>(fx * 65536);
  Halide::Expr xi = Halide::cast<int>(x * fx / 65536);
  Halide::Expr xr = Halide::cast<uint16_t>(x * fx % 65536);
  Halide::Expr x0 = input(xi + 0, y, _);
  Halide::Expr x1 = input(xi + 1, y, _);
  x_interpolated(x, y, _) = lerp(x0, x1, xr);

  fy = Halide::cast<int>(fy * 65536);
  Halide::Expr yi = Halide::cast<int>(y * fy / 65536);
  Halide::Expr yr = Halide::cast<uint16_t>(y * fy % 65536);
  Halide::Expr y0 = x_interpolated(x, yi + 0, _);
  Halide::Expr y1 = x_interpolated(x, yi + 1, _);
  result(x, y, _) = lerp(y0, y1, yr);
}

void rotate(Halide::Func input, Halide::Func result, Halide::Expr width,
            Halide::Expr height, Halide::Expr angle) {
  Halide::Var x{"x"}, y{"y"};
  Halide::Func result_90_degrees, result_180_degrees, result_270_degrees;
  result_90_degrees(x, y, _) = input(width - 1 - y, x, _);
  result_180_degrees(x, y, _) = input(width - 1 - x, height - 1 - y, _);
  result_270_degrees(x, y, _) = input(y, height - 1 - x, _);

  result(x, y, _) =
      select(angle == 90, result_90_degrees(x, y, _), angle == 180,
             result_180_degrees(x, y, _), angle == 270,
             result_270_degrees(x, y, _), input(x, y, _));
}

}  // namespace common
}  // namespace halide
}  // namespace frame_buffer
}  // namespace mediapipe
