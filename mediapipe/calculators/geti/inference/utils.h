#ifndef UTILS_H_
#define UTILS_H_

#include <string>

#include "mediapipe/framework/calculator_framework.h"
#include "../utils/data_structures.h"
#include "utils/ocv_common.hpp"

namespace geti {

extern const std::string GETI_EMPTY_LABEL;

static inline std::vector<Label> get_labels_from_configuration(
    ov::AnyMap configuration) {
  auto labels_iter = configuration.find("labels");
  auto label_ids_iter = configuration.find("label_ids");
  std::vector<Label> labels = {};
  if (labels_iter != configuration.end() &&
      label_ids_iter != configuration.end()) {
    std::vector<std::string> label_ids =
        label_ids_iter->second.as<std::vector<std::string>>();
    std::vector<std::string> label_names =
        labels_iter->second.as<std::vector<std::string>>();
    for (size_t i = 0; i < label_ids.size(); i++) {
      if (label_names.size() > i)
        labels.push_back({label_ids[i], label_names[i]});
      else
        labels.push_back({label_ids[i], ""});
    }
  }
  return labels;
}

static inline std::string get_output_tag(std::string tag,
                                         std::vector<std::string> fallbacks,
                                         mediapipe::CalculatorContext* cc) {
  for (std::string& value : fallbacks) {
    if (cc->Outputs().HasTag(value)) {
      LOG(WARNING)
          << "DEPRICATION: Graph is using 1.13 interface for calculators";
      return value;
    }
  }
  return tag;
}

static inline std::string get_input_tag(std::string tag,
                                        std::vector<std::string> fallbacks,
                                        mediapipe::CalculatorContext* cc) {
  for (std::string& value : fallbacks) {
    if (cc->Inputs().HasTag(value)) {
      LOG(WARNING)
          << "DEPRICATION: Graph is using 1.13 interface for calculators";
      return value;
    }
  }
  return tag;
}

static inline cv::Mat get_mat_from_ov_tensor(ov::Tensor& tensor,
                                             size_t& shape_shift,
                                             size_t layer) {
  // The cv::Mat constructor in wrap_saliency_map_tensor_to_mat doesn't copy
  // over the data. This means that once the inference_result is cleaned up
  // the memory might be overwritten. A clone makes sure that the memory is
  // held by the cv::Mat.

  return wrap_saliency_map_tensor_to_mat(tensor, shape_shift, layer).clone();
}

}  // namespace geti

#endif  // UTILS_H_
