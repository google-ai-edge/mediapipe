#ifndef RESULT_SERIALIZATION_H_
#define RESULT_SERIALIZATION_H_

#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "nlohmann/json.hpp"
#include "third_party/cpp-base64/base64.h"
#include "../utils/data_structures.h"

namespace cv {
inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const cv ::Point& nlohmann_json_t) {
  nlohmann_json_j["x"] = nlohmann_json_t.x;
  nlohmann_json_j["y"] = nlohmann_json_t.y;
}

inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const cv ::Rect& nlohmann_json_t) {
  nlohmann_json_j["x"] = nlohmann_json_t.x;
  nlohmann_json_j["y"] = nlohmann_json_t.y;
  nlohmann_json_j["width"] = nlohmann_json_t.width;
  nlohmann_json_j["height"] = nlohmann_json_t.height;
  nlohmann_json_j["type"] = "RECTANGLE";
}

inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const cv ::RotatedRect& nlohmann_json_t) {
  nlohmann_json_j["x"] = nlohmann_json_t.center.x;
  nlohmann_json_j["y"] = nlohmann_json_t.center.y;
  nlohmann_json_j["width"] = nlohmann_json_t.size.width;
  nlohmann_json_j["height"] = nlohmann_json_t.size.height;
  nlohmann_json_j["angle"] = nlohmann_json_t.angle;
  nlohmann_json_j["type"] = "ROTATED_RECTANGLE";
}
}  // namespace cv

namespace geti {

static inline std::string base64_encode_mat(cv::Mat image) {
  std::vector<uchar> buf;
  if (!image.empty()) cv::imencode(".jpg", image, buf);
  auto* enc_msg = reinterpret_cast<unsigned char*>(buf.data());
  return base64_encode(enc_msg, buf.size());
}

inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const SaliencyMap& nlohmann_json_t) {
  nlohmann_json_j["data"] = base64_encode_mat(nlohmann_json_t.image);
  nlohmann_json_j["label_id"] = nlohmann_json_t.label.label_id;
}

inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const LabelResult& nlohmann_json_t) {
  nlohmann_json_j["probability"] = nlohmann_json_t.probability;
  nlohmann_json_j["id"] = nlohmann_json_t.label.label_id;
}
inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const PolygonPrediction& nlohmann_json_t) {
  nlohmann_json_j["labels"] = nlohmann_json_t.labels;
  nlohmann_json_j["shape"]["points"] = nlohmann_json_t.shape;
  nlohmann_json_j["shape"]["type"] = "POLYGON";
}

inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const RectanglePrediction& nlohmann_json_t) {
  nlohmann_json_j["labels"] = nlohmann_json_t.labels;
  nlohmann_json_j["shape"] = nlohmann_json_t.shape;
}

inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const RotatedRectanglePrediction& nlohmann_json_t) {
  nlohmann_json_j["labels"] = nlohmann_json_t.labels;
  nlohmann_json_j["shape"] = nlohmann_json_t.shape;
}

inline void to_json(nlohmann ::json& nlohmann_json_j,
                    const InferenceResult& nlohmann_json_t) {
  nlohmann::json rects = nlohmann_json_t.rectangles;
  nlohmann::json rotated_rects = nlohmann_json_t.rotated_rectangles;
  nlohmann::json polygons = nlohmann_json_t.polygons;
  auto predictions = nlohmann::json::array();
  predictions.insert(predictions.end(), rects.begin(), rects.end());
  predictions.insert(predictions.end(), rotated_rects.begin(),
                     rotated_rects.end());
  predictions.insert(predictions.end(), polygons.begin(), polygons.end());
  nlohmann_json_j["predictions"] = predictions;
  nlohmann_json_j["maps"] = nlohmann_json_t.saliency_maps;
}

static inline void filter_maps_by_prediction_prevalence(nlohmann::json& data) {
  if (!data.contains("maps")) {
    return;
  }
  std::map<std::string, bool> label_prevalence_map;
  for (auto& prediction : data["predictions"]) {
    for (auto& label : prediction["labels"]) {
      label_prevalence_map[label["id"]] = true;
    }
  }

  for (auto element = data["maps"].begin(); element != data["maps"].end();) {
    auto label_id = element.value()["label_id"];
    if (label_prevalence_map.find(label_id) == label_prevalence_map.end()) {
      element = data["maps"].erase(element);
    } else {
      ++element;
    }
  }
}

static inline void translate_inference_result_by_roi(InferenceResult& result,
                                                     int roi_x, int roi_y) {
  if (roi_x == 0 && roi_y == 0) {
    return;
  }

  for (auto& polygon : result.polygons) {
    for (auto& point : polygon.shape) {
      point.x += roi_x;
      point.y += roi_y;
    }
  }

  for (auto& rect : result.rectangles) {
    rect.shape.x += roi_x;
    rect.shape.y += roi_y;
  }

  for (auto& rect : result.rotated_rectangles) {
    rect.shape.center.x += roi_x;
    rect.shape.center.y += roi_y;
  }
}

}  // namespace geti

#endif  // RESULT_SERIALIZATION_H_
