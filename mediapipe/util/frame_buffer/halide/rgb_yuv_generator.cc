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

class RgbYuv : public Halide::Generator<RgbYuv> {
 public:
  Var x{"x"}, y{"y"}, c{"c"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 3>> src_rgb{"rgb"};

  Output<Func> dst_y{"dst_y", UInt(8), 2};
  Output<Func> dst_uv{"dst_uv", UInt(8), 3};

  void generate();
  void schedule();
};

// Integer math versions of the full-range JFIF RGB-YUV coefficients.
//   Y =  0.2990*R + 0.5870*G + 0.1140*B
//   U = -0.1687*R - 0.3313*G + 0.5000*B + 128
//   V =  0.5000*R - 0.4187*G - 0.0813*B + 128
// See https://www.w3.org/Graphics/JPEG/jfif3.pdf. These coefficients are
// similar to, but not identical, to those used in Android.
Halide::Tuple rgbyuv(Halide::Expr r, Halide::Expr g, Halide::Expr b) {
  r = Halide::cast<int32_t>(r);
  g = Halide::cast<int32_t>(g);
  b = Halide::cast<int32_t>(b);
  return {
      (19595 * r + 38470 * g + 7474 * b + 32768) >> 16,
      ((-11056 * r - 21712 * g + 32768 * b + 32768) >> 16) + 128,
      ((32768 * r - 27440 * g - 5328 * b + 32768) >> 16) + 128,
  };
}

void RgbYuv::generate() {
  Halide::Func yuv_tuple("yuv_tuple");
  yuv_tuple(x, y) =
      rgbyuv(src_rgb(x, y, 0), src_rgb(x, y, 1), src_rgb(x, y, 2));

  // Y values are copied one-for-one; UV values are sampled 1/4.
  // TODO: Take the average UV values across the 2x2 block.
  dst_y(x, y) = Halide::saturating_cast<uint8_t>(yuv_tuple(x, y)[0]);
  dst_uv(x, y, c) = Halide::saturating_cast<uint8_t>(Halide::select(
      c == 0, yuv_tuple(x * 2, y * 2)[2], yuv_tuple(x * 2, y * 2)[1]));
  // NOTE: uv channel indices above assume NV21; this can be abstracted out
  // by twiddling strides in calling code.
}

void RgbYuv::schedule() {
  // RGB images starts at index zero in every dimension.
  src_rgb.dim(0).set_min(0);
  src_rgb.dim(1).set_min(0);
  src_rgb.dim(2).set_min(0);

  // Require that the input buffer be interleaved and tightly-packed;
  // that is, either RGBRGBRGB[...] or RGBARGBARGBA[...], without gaps
  // between pixels.
  src_rgb.dim(0).set_stride(src_rgb.dim(2).extent());
  src_rgb.dim(2).set_stride(1);

  // Y plane dimensions start at zero. We could additionally constrain the
  // extent to be even, but that doesn't seem to have any benefit.
  Halide::Func dst_y_func = dst_y;
  Halide::OutputImageParam dst_y_output = dst_y_func.output_buffer();
  dst_y_output.dim(0).set_min(0);
  dst_y_output.dim(1).set_min(0);

  // UV plane has two channels and is half the size of the Y plane in X/Y.
  Halide::Func dst_uv_func = dst_uv;
  Halide::OutputImageParam dst_uv_output = dst_uv_func.output_buffer();
  dst_uv_output.dim(0).set_bounds(0, (dst_y_output.dim(0).extent() + 1) / 2);
  dst_uv_output.dim(1).set_bounds(0, (dst_y_output.dim(1).extent() + 1) / 2);
  dst_uv_output.dim(2).set_bounds(0, 2);

  // UV channel processing should be loop unrolled.
  dst_uv_func.reorder(c, x, y);
  dst_uv_func.unroll(c);

  // Remove default memory layout constraints and accept/produce generic UV
  // (including semi-planar and planar).
  dst_uv_output.dim(0).set_stride(Expr());
}

}  // namespace

HALIDE_REGISTER_GENERATOR(RgbYuv, rgb_yuv_generator)
