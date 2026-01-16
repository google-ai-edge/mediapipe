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

using ::Halide::BoundaryConditions::repeat_edge;
using ::mediapipe::frame_buffer::halide::common::resize_bilinear_int;

class GrayResize : public Halide::Generator<GrayResize> {
 public:
  Var x{"x"}, y{"y"};

  Input<Buffer<uint8_t, 2>> src_y{"src_y"};
  Input<float> scale_x{"scale_x", 1.0f, 0.0f, 1024.0f};
  Input<float> scale_y{"scale_y", 1.0f, 0.0f, 1024.0f};

  Output<Func> dst_y{"dst_y", UInt(8), 2};

  void generate();
  void schedule();
};

void GrayResize::generate() {
  resize_bilinear_int(repeat_edge(src_y), dst_y, scale_x, scale_y);
}

void GrayResize::schedule() {
  // Grayscale image dimensions start at zero.
  Halide::Func dst_y_func = dst_y;
  Halide::OutputImageParam dst_y_output = dst_y_func.output_buffer();
  src_y.dim(0).set_min(0);
  src_y.dim(1).set_min(0);
  dst_y_output.dim(0).set_min(0);
  dst_y_output.dim(1).set_min(0);

  // We must ensure that the image is wide enough to support vector
  // operations.
  const int vector_size = natural_vector_size<uint8_t>();
  Halide::Expr min_y_width =
      Halide::min(src_y.dim(0).extent(), dst_y_output.dim(0).extent());
  dst_y_func.specialize(min_y_width >= vector_size).vectorize(x, vector_size);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(GrayResize, gray_resize_generator)
