/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023 Intel Corporation
 *
 *  This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 *  This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */
#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <models/results.h>

#include <vector>

namespace geti {
struct Label {
  std::string label_id;
  std::string label;
};

struct SaliencyMap {
  cv::Mat image;
  cv::Rect roi;
  Label label;
};

struct LabelResult {
  float probability;
  Label label;
};

struct PolygonPrediction {
  std::vector<LabelResult> labels;
  std::vector<cv::Point> shape;
};

struct RectanglePrediction {
  std::vector<LabelResult> labels;
  cv::Rect shape;
};

struct RotatedRectanglePrediction {
  std::vector<LabelResult> labels;
  cv::RotatedRect shape;
};

struct InferenceResult {
  std::vector<RectanglePrediction> rectangles;
  std::vector<RotatedRectanglePrediction> rotated_rectangles;
  std::vector<PolygonPrediction> polygons;
  std::vector<SaliencyMap> saliency_maps;
  cv::Rect roi;
};
}  // namespace geti

#endif  // DATA_STRUCTURES_H
