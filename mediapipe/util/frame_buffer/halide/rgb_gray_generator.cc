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

class RgbGray : public Halide::Generator<RgbGray> {
 public:
  Var x{"x"}, y{"y"}, c{"c"};

  Input<Buffer<uint8_t, 3>> src_rgb{"rgb"};
  Output<Buffer<uint8_t, 2>> convert{"convert"};

  void generate();
  void schedule();
};

// Integer math versions of the full-range JFIF RGB-Y coefficients.
//   Y =  0.2990*R + 0.5870*G + 0.1140*B
// See https://www.w3.org/Graphics/JPEG/jfif3.pdf. These coefficients are
// similar to, but not identical, to those used in Android.
Halide::Expr rgby(Halide::Expr r, Halide::Expr g, Halide::Expr b) {
  r = Halide::cast<int32_t>(r);
  g = Halide::cast<int32_t>(g);
  b = Halide::cast<int32_t>(b);
  return (19595 * r + 38470 * g + 7474 * b + 32768) >> 16;
}

void RgbGray::generate() {
  Halide::Func gray("gray");
  gray(x, y) = rgby(src_rgb(x, y, 0), src_rgb(x, y, 1), src_rgb(x, y, 2));
  convert(x, y) = Halide::saturating_cast<uint8_t>(gray(x, y));
}

void RgbGray::schedule() {
  // RGB images starts at index zero in every dimension.
  src_rgb.dim(0).set_min(0);
  src_rgb.dim(1).set_min(0);
  src_rgb.dim(2).set_min(0);

  // Require that the input buffer be interleaved and tightly-packed;
  // with no gaps between pixels.
  src_rgb.dim(0).set_stride(src_rgb.dim(2).extent());
  src_rgb.dim(2).set_stride(1);

  // Grayscale images starts at index zero in every dimension.
  convert.dim(0).set_min(0);
  convert.dim(1).set_min(0);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(RgbGray, rgb_gray_generator)
