// Copyright 2019 The MediaPipe Authors.
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

#include <cmath>
#include <vector>

#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

float CalculateScale(float min_scale, float max_scale, int stride_index,
                     int num_strides) {
  if (num_strides == 1) {
    return (min_scale + max_scale) * 0.5f;
  } else {
    return min_scale +
           (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
  }
}

}  // namespace

// Generate anchors for SSD object detection model.
// Output:
//   ANCHORS: A list of anchors. Model generates predictions based on the
//   offsets of these anchors.
//
// Usage example:
// node {
//   calculator: "SsdAnchorsCalculator"
//   output_side_packet: "anchors"
//   options {
//     [mediapipe.SsdAnchorsCalculatorOptions.ext] {
//       num_layers: 6
//       min_scale: 0.2
//       max_scale: 0.95
//       input_size_height: 300
//       input_size_width: 300
//       anchor_offset_x: 0.5
//       anchor_offset_y: 0.5
//       strides: 16
//       strides: 32
//       strides: 64
//       strides: 128
//       strides: 256
//       strides: 512
//       aspect_ratios: 1.0
//       aspect_ratios: 2.0
//       aspect_ratios: 0.5
//       aspect_ratios: 3.0
//       aspect_ratios: 0.3333
//       reduce_boxes_in_lowest_layer: true
//     }
//   }
// }
class SsdAnchorsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->OutputSidePackets().Index(0).Set<std::vector<Anchor>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    const SsdAnchorsCalculatorOptions& options =
        cc->Options<SsdAnchorsCalculatorOptions>();

    auto anchors = absl::make_unique<std::vector<Anchor>>();
    MP_RETURN_IF_ERROR(GenerateAnchors(anchors.get(), options));
    cc->OutputSidePackets().Index(0).Set(Adopt(anchors.release()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  static absl::Status GenerateAnchors(
      std::vector<Anchor>* anchors, const SsdAnchorsCalculatorOptions& options);
};
REGISTER_CALCULATOR(SsdAnchorsCalculator);

absl::Status SsdAnchorsCalculator::GenerateAnchors(
    std::vector<Anchor>* anchors, const SsdAnchorsCalculatorOptions& options) {
  // Verify the options.
  if (!options.feature_map_height_size() && !options.strides_size()) {
    return absl::InvalidArgumentError(
        "Both feature map shape and strides are missing. Must provide either "
        "one.");
  }
  if (options.feature_map_height_size()) {
    if (options.strides_size()) {
      LOG(ERROR) << "Found feature map shapes. Strides will be ignored.";
    }
    CHECK_EQ(options.feature_map_height_size(), options.num_layers());
    CHECK_EQ(options.feature_map_height_size(),
             options.feature_map_width_size());
  } else {
    CHECK_EQ(options.strides_size(), options.num_layers());
  }

  int layer_id = 0;
  while (layer_id < options.num_layers()) {
    std::vector<float> anchor_height;
    std::vector<float> anchor_width;
    std::vector<float> aspect_ratios;
    std::vector<float> scales;

    // For same strides, we merge the anchors in the same order.
    int last_same_stride_layer = layer_id;
    while (last_same_stride_layer < options.strides_size() &&
           options.strides(last_same_stride_layer) ==
               options.strides(layer_id)) {
      const float scale =
          CalculateScale(options.min_scale(), options.max_scale(),
                         last_same_stride_layer, options.strides_size());
      if (last_same_stride_layer == 0 &&
          options.reduce_boxes_in_lowest_layer()) {
        // For first layer, it can be specified to use predefined anchors.
        aspect_ratios.push_back(1.0);
        aspect_ratios.push_back(2.0);
        aspect_ratios.push_back(0.5);
        scales.push_back(0.1);
        scales.push_back(scale);
        scales.push_back(scale);
      } else {
        for (int aspect_ratio_id = 0;
             aspect_ratio_id < options.aspect_ratios_size();
             ++aspect_ratio_id) {
          aspect_ratios.push_back(options.aspect_ratios(aspect_ratio_id));
          scales.push_back(scale);
        }
        if (options.interpolated_scale_aspect_ratio() > 0.0) {
          const float scale_next =
              last_same_stride_layer == options.strides_size() - 1
                  ? 1.0f
                  : CalculateScale(options.min_scale(), options.max_scale(),
                                   last_same_stride_layer + 1,
                                   options.strides_size());
          scales.push_back(std::sqrt(scale * scale_next));
          aspect_ratios.push_back(options.interpolated_scale_aspect_ratio());
        }
      }
      last_same_stride_layer++;
    }

    for (int i = 0; i < aspect_ratios.size(); ++i) {
      const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
      anchor_height.push_back(scales[i] / ratio_sqrts);
      anchor_width.push_back(scales[i] * ratio_sqrts);
    }

    int feature_map_height = 0;
    int feature_map_width = 0;
    if (options.feature_map_height_size()) {
      feature_map_height = options.feature_map_height(layer_id);
      feature_map_width = options.feature_map_width(layer_id);
    } else {
      const int stride = options.strides(layer_id);
      feature_map_height =
          std::ceil(1.0f * options.input_size_height() / stride);
      feature_map_width = std::ceil(1.0f * options.input_size_width() / stride);
    }

    for (int y = 0; y < feature_map_height; ++y) {
      for (int x = 0; x < feature_map_width; ++x) {
        for (int anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
          // TODO: Support specifying anchor_offset_x, anchor_offset_y.
          const float x_center =
              (x + options.anchor_offset_x()) * 1.0f / feature_map_width;
          const float y_center =
              (y + options.anchor_offset_y()) * 1.0f / feature_map_height;

          Anchor new_anchor;
          new_anchor.set_x_center(x_center);
          new_anchor.set_y_center(y_center);

          if (options.fixed_anchor_size()) {
            new_anchor.set_w(1.0f);
            new_anchor.set_h(1.0f);
          } else {
            new_anchor.set_w(anchor_width[anchor_id]);
            new_anchor.set_h(anchor_height[anchor_id]);
          }
          anchors->push_back(new_anchor);
        }
      }
    }
    layer_id = last_same_stride_layer;
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
