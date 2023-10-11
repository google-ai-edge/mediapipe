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

class YuvRgb : public Halide::Generator<YuvRgb> {
 public:
  Var x{"x"}, y{"y"}, c{"c"};

  // Input<Buffer> because that allows us to apply constraints on stride, etc.
  Input<Buffer<uint8_t, 2>> src_y{"src_y"};
  Input<Buffer<uint8_t, 3>> src_uv{"src_uv"};
  Input<bool> halve{"halve", false};

  Output<Func> rgb{"rgb", UInt(8), 3};

  void generate();
  void schedule();
};

Halide::Expr demux(Halide::Expr c, Halide::Tuple values) {
  return select(c == 0, values[0], c == 1, values[1], c == 2, values[2], 255);
}

// Integer math versions of the full-range JFIF YUV-RGB coefficients.
//   R = Y' + 1.40200*(V-128)
//   G = Y' - 0.34414*(U-128) - 0.71414*(V-128)
//   B = Y' + 1.77200*(U-128)
// See https://www.w3.org/Graphics/JPEG/jfif3.pdf. These coefficients are
// similar to, but not identical, to those used in Android.
Halide::Tuple yuvrgb(Halide::Expr y, Halide::Expr u, Halide::Expr v) {
  y = Halide::cast<int32_t>(y);
  u = Halide::cast<int32_t>(u) - 128;
  v = Halide::cast<int32_t>(v) - 128;
  return {
      y + ((91881 * v + 32768) >> 16),
      y - ((22544 * u + 46802 * v + 32768) >> 16),
      y + ((116130 * u + 32768) >> 16),
  };
}

void YuvRgb::generate() {
  // Each 2x2 block of Y pixels shares the same UV values, so UV-coordinates
  // advance half as slowly as Y-coordinates. When taking advantage of the
  // "free" 2x downsampling, use every UV value but skip every other Y.
  Halide::Expr yx = select(halve, 2 * x, x), yy = select(halve, 2 * y, y);
  Halide::Expr uvx = select(halve, x, x / 2), uvy = select(halve, y, y / 2);

  rgb(x, y, c) = Halide::saturating_cast<uint8_t>(demux(
      c, yuvrgb(src_y(yx, yy), src_uv(uvx, uvy, 1), src_uv(uvx, uvy, 0))));
  // NOTE: uv channel indices above assume NV21; this can be abstracted out
  // by twiddling strides in calling code.
}

void YuvRgb::schedule() {
  // Y plane dimensions start at zero. We could additionally constrain the
  // extent to be even, but that doesn't seem to have any benefit.
  src_y.dim(0).set_min(0);
  src_y.dim(1).set_min(0);

  // UV plane has two channels and is half the size of the Y plane in X/Y.
  src_uv.dim(0).set_bounds(0, (src_y.dim(0).extent() + 1) / 2);
  src_uv.dim(1).set_bounds(0, (src_y.dim(1).extent() + 1) / 2);
  src_uv.dim(2).set_bounds(0, 2);

  // Remove default memory layout constraints on the UV source so that we
  // accept generic UV (including semi-planar and planar).
  //
  // TODO: Investigate whether it's worth specializing the cross-
  // product of [semi-]planar and RGB/RGBA; this would result in 9 codepaths.
  src_uv.dim(0).set_stride(Expr());

  Halide::Func rgb_func = rgb;
  Halide::OutputImageParam rgb_output = rgb_func.output_buffer();
  Halide::Expr rgb_channels = rgb_output.dim(2).extent();

  // Specialize the generated code for RGB and RGBA.
  const int vector_size = natural_vector_size<uint8_t>();
  rgb_func.reorder(c, x, y);
  rgb_func.specialize(rgb_channels == 3).unroll(c).vectorize(x, vector_size);
  rgb_func.specialize(rgb_channels == 4).unroll(c).vectorize(x, vector_size);

  // Require that the output buffer be interleaved and tightly-packed;
  // that is, either RGBRGBRGB[...] or RGBARGBARGBA[...], without gaps
  // between pixels.
  rgb_output.dim(0).set_stride(rgb_channels);
  rgb_output.dim(2).set_stride(1);

  // RGB output starts at index zero in every dimension.
  rgb_output.dim(0).set_min(0);
  rgb_output.dim(1).set_min(0);
  rgb_output.dim(2).set_min(0);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(YuvRgb, yuv_rgb_generator)
