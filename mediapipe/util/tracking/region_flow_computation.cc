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

#include "mediapipe/util/tracking/region_flow_computation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>

#include "Eigen/Core"
#include "absl/container/node_hash_set.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_features2d_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/tracking/camera_motion.pb.h"
#include "mediapipe/util/tracking/image_util.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/motion_estimation.h"
#include "mediapipe/util/tracking/motion_estimation.pb.h"
#include "mediapipe/util/tracking/motion_models.h"
#include "mediapipe/util/tracking/parallel_invoker.h"
#include "mediapipe/util/tracking/region_flow.h"
#include "mediapipe/util/tracking/tone_estimation.h"
#include "mediapipe/util/tracking/tone_estimation.pb.h"
#include "mediapipe/util/tracking/tone_models.h"
#include "mediapipe/util/tracking/tone_models.pb.h"

using std::max;
using std::min;

namespace mediapipe {

typedef RegionFlowFrame::RegionFlow RegionFlow;
typedef RegionFlowFeature Feature;
constexpr float kZeroMotion = 0.25f;  // Quarter pixel average motion.

// Helper struct used by RegionFlowComputation and MotionEstimation.
// Feature position, flow and error. Unique id per track, set to -1 if no such
// id can be assigned.
struct TrackedFeature {
  TrackedFeature(const Vector2_f& point_, const Vector2_f& flow_,
                 float tracking_error_, float corner_response_, int octave_,
                 int track_id_ = -1, float verify_dist_ = 0)
      : point(point_),
        flow(flow_),
        tracking_error(tracking_error_),
        corner_response(corner_response_),
        octave(octave_),
        irls_weight(1.0f),
        num_bins(1),
        track_id(track_id_),
        verify_dist(verify_dist_) {}

  Vector2_f point;
  Vector2_f flow;
  float tracking_error = 0;
  float corner_response = 0;
  int octave = 0;
  float irls_weight = 1.0f;
  int num_bins = 1;   // Total number of bins feature is binned into.
  int track_id = -1;  // Unique id, assigned to each feature belonging to the
                      // same track. Negative values indicates no id.
  float verify_dist = 0;

  // Flags as defined by RegionFlowFeature.
  int flags = 0;

  // Descriptors of this feature (single row).
  cv::Mat descriptors;

  // Original neighborhood of the feature. Refers to the patch that the feature
  // was extracted the very first time.
  // Using shared_ptr instead of unique_ptr to make TrackingData copyable.
  // We don't use cv::Mat directly (which is reference counted) as
  // neighborhoods are only used for long feature verification (which is
  // optional).
  std::shared_ptr<const cv::Mat> orig_neighborhood;

  void Invert() {
    point += flow;
    flow = -flow;
  }
};

// Inverts features (swaps location and matches). In-place operation OK, i.e.
// inverted_list == &list acceptable and faster.
void InvertFeatureList(const TrackedFeatureList& list,
                       TrackedFeatureList* inverted_list) {
  if (inverted_list != &list) {
    *inverted_list = list;
  }

  for (auto& feature : *inverted_list) {
    feature.Invert();
  }
}

// Allocates pyramid images of sufficient size (suggested OpenCV settings,
// independent of number of pyramid levels).
void AllocatePyramid(int frame_width, int frame_height, cv::Mat* pyramid) {
  const int pyramid_width = frame_width + 8;
  const int pyramid_height = frame_height / 2 + 1;
  pyramid->create(pyramid_height, pyramid_width, CV_8UC1);
}

namespace {

// lab_window is used as scratch space only, to avoid allocations.
void GetPatchDescriptorAtPoint(const cv::Mat& rgb_frame, const Vector2_i& pt,
                               const int radius, cv::Mat* lab_window,
                               PatchDescriptor* descriptor) {
  CHECK(descriptor);
  descriptor->clear_data();

  // Reserve enough data for mean and upper triangular part of
  // covariance matrix.
  descriptor->mutable_data()->Reserve(3 + 6);

  // Extract a window of the RGB frame for Lab color conversion. We know that at
  // this point the window doesn't overlap with the frame boundary. The
  // windowing operation just generates a reference and doesn't copy the values.
  const int diameter = 2 * radius + 1;
  const cv::Mat rgb_window =
      rgb_frame(cv::Rect(pt.x() - radius, pt.y() - radius, diameter, diameter));

  // Compute channel sums and means.
  int sum[3] = {0, 0, 0};
  for (int y = 0; y < diameter; ++y) {
    const uint8* data = rgb_window.ptr<uint8>(y);
    for (int x = 0; x < diameter; ++x, data += 3) {
      for (int c = 0; c < 3; ++c) {
        sum[c] += data[c];
      }
    }
  }
  const float scale = 1.f / (diameter * diameter);
  for (int c = 0; c < 3; ++c) {
    descriptor->add_data(sum[c] * scale);  // Mean value.
  }

  const float denom = 1.0f / (diameter * diameter);

  // Compute the channel dot products, after centering around the respective
  // channel means. Only computing upper triangular part.
  int product[3][3];
  for (int c = 0; c < 3; ++c) {
    for (int d = c; d < 3; ++d) {
      // We want to compute
      //     sum_{x,y}[(data[c] - mean[c]) * (data[d] - mean[d])],
      // which simplifies to
      //     sum_{x,y}[data[c] * data[d]] - sum[c] * sum[d] / N
      // using N = diameter * diameter and sum[c] = N * mean[c].
      product[c][d] = -sum[c] * sum[d] * denom;
      for (int y = 0; y < diameter; ++y) {
        const uint8* data = rgb_window.ptr<uint8>(y);
        for (int x = 0; x < diameter; ++x, data += 3) {
          product[c][d] += static_cast<int>(data[c]) * data[d];
        }
      }
    }
  }

  // Finally, add the descriptors only storring upper triangular part.
  for (int c = 0; c < 3; ++c) {
    for (int d = c; d < 3; ++d) {
      descriptor->add_data(product[c][d] * scale);
    }
  }
}

class PatchDescriptorInvoker {
 public:
  PatchDescriptorInvoker(const cv::Mat& rgb_frame,
                         const cv::Mat* prev_rgb_frame, int radius,
                         RegionFlowFeatureList* features)
      : rgb_frame_(rgb_frame),
        prev_rgb_frame_(prev_rgb_frame),
        radius_(radius),
        features_(features) {}

  void operator()(const BlockedRange& range) const {
    cv::Mat lab_window;  // To avoid repeated allocations below.
    for (int feature_idx = range.begin(); feature_idx != range.end();
         ++feature_idx) {
      RegionFlowFeature* feature = features_->mutable_feature(feature_idx);
      Vector2_i pt(FeatureIntLocation(*feature));
      DCHECK_GE(pt.x(), radius_);
      DCHECK_GE(pt.y(), radius_);
      DCHECK_LT(pt.x(), rgb_frame_.cols - radius_);
      DCHECK_LT(pt.y(), rgb_frame_.rows - radius_);
      GetPatchDescriptorAtPoint(rgb_frame_, pt, radius_, &lab_window,
                                feature->mutable_feature_descriptor());

      if (prev_rgb_frame_) {
        Vector2_i pt_match(FeatureMatchIntLocation(*feature));
        DCHECK_GE(pt_match.x(), radius_);
        DCHECK_GE(pt_match.y(), radius_);
        DCHECK_LT(pt_match.x(), rgb_frame_.cols - radius_);
        DCHECK_LT(pt_match.y(), rgb_frame_.rows - radius_);
        GetPatchDescriptorAtPoint(*prev_rgb_frame_, pt_match, radius_,
                                  &lab_window,
                                  feature->mutable_feature_match_descriptor());
      }
    }
  }

 private:
  const cv::Mat& rgb_frame_;
  const cv::Mat* prev_rgb_frame_;
  int radius_;
  RegionFlowFeatureList* features_;
};

}  // namespace.

// Computes patch descriptor in color domain (LAB), see region_flow.proto for
// specifics.
// If optional parameter prev_rgb_frame is set, also computes corresponding
// feature_match_descriptor.
// IMPORTANT: Ensure that patch_descriptor_rad <= distance_from_border in
// GetRegionFlowFeatureList. Checked by function.
void ComputeRegionFlowFeatureDescriptors(
    const cv::Mat& rgb_frame, const cv::Mat* prev_rgb_frame,
    int patch_descriptor_radius, RegionFlowFeatureList* flow_feature_list) {
  const int rows = rgb_frame.rows;
  const int cols = rgb_frame.cols;
  CHECK_EQ(rgb_frame.depth(), CV_8U);
  CHECK_EQ(rgb_frame.channels(), 3);

  if (prev_rgb_frame) {
    CHECK_EQ(prev_rgb_frame->depth(), CV_8U);
    CHECK_EQ(prev_rgb_frame->channels(), 3);
    CHECK_EQ(prev_rgb_frame->rows, rows);
    CHECK_EQ(prev_rgb_frame->cols, cols);
  }

  CHECK_LE(patch_descriptor_radius, flow_feature_list->distance_from_border());

  ParallelFor(
      0, flow_feature_list->feature_size(), 1,
      PatchDescriptorInvoker(rgb_frame, prev_rgb_frame, patch_descriptor_radius,
                             flow_feature_list));
}

// Stores 2D location's of feature points and their corresponding descriptors,
// where the i'th row in descriptors corresponds to the i'th entry in
// key_points.
struct RegionFlowComputation::ORBFeatureDescriptors {
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> key_points;
  bool computed;

  ORBFeatureDescriptors() { Reset(); }

  void Reset() {
    key_points.clear();
    computed = false;
  }
};

struct RegionFlowComputation::FrameTrackingData {
  cv::Mat frame;

  // Pyramid used for tracking. Just contains the a single image if old
  // c-interface is used.
  std::vector<cv::Mat> pyramid;
  cv::Mat blur_data;
  cv::Mat tiny_image;  // Used if visual consistency verification is performed.
  cv::Mat mask;  // Features need to be extracted only where mask value > 0.
  float mean_intensity = 0;  // Mean intensity of the frame.

  // Pyramid used during feature extraction at multiple levels.
  std::vector<cv::Mat> extraction_pyramid;

  // Records number of pyramid levels stored by member pyramid. If zero, pyramid
  // has not been computed yet.
  int pyramid_levels = 0;

  // Features extracted in this frame or tracked from a source frame.
  std::vector<cv::Point2f> features;

  // FrameTrackingData that resulting features were tracked from.
  FrameTrackingData* source = nullptr;

  // Indicates for each feature, corresponding source feature index (index into
  // above features vector for source FrameTrackingData).
  std::vector<int> feature_source_map;

  // If set, indicates that member features was pre-initialized.
  bool features_initialized = false;

  // Time (in frames) when the last feature extraction was carried out for the
  // features used in this frame. Time increases by 1 every time we reuse
  // tracked features as the features for this frame.
  int last_feature_extraction_time = -1;

  // Number of extracted and tracked features in the original extraction frame
  // (referred to by last_feature_extraction_time) and the current frame.
  // That is stores the number inliers according to function
  // GetFeatureTrackInliers.
  int num_original_extracted_and_tracked = -1;  // Negative for not set.
  int num_extracted_and_tracked = -1;           // Negative for not set.

  // 1:1 mapping w.r.t. features.
  std::vector<float> corner_responses;

  // 1:1 mapping w.r.t. features. Records octave each feature belongs to.
  std::vector<int> octaves;

  // 1:1 mapping w.r.t. features. Records track id each feature belongs to,
  // use -1 if no such track id exists. Only used for long feature tracks.
  std::vector<int> track_ids;

  // Stores all the tracked ids that has been discarded actively in this frame.
  // This information will be popluated via RegionFlowFeatureList, so that the
  // downstreaming modules can receive it and use it to avoid misjudgement on
  // tracking continuity.
  // Discard reason:
  // (1) A tracked feature has too long track, which might create drift.
  // (2) A tracked feature in a highly densed area, which provides littel value.
  std::vector<int> actively_discarded_tracked_ids;

  // 1:1 mapping w.r.t. features. Stores the original patch neighborhood when
  // the feature was extracted for the first time, that is for features with
  // assigned track_id (>= 0) the data refers to a neighborhood in an earlier
  // frame or else the neighborhood in the current frame.
  // Stores a CV_8U grayscale square patch with length
  // TrackingOptions::tracking_window_size().
  std::shared_ptr<std::vector<cv::Mat>> neighborhoods;

  // Absolute frame number of this FrameTrackingData.
  int frame_num = 0;

  // Timestamp of the underlying frame.
  int64 timestamp_usec = 0;

  // Difference of this FrameTrackingData's tiny_image w.r.t. previous one,
  // i.e. one frame earlier.
  float tiny_image_diff = 0.0f;

  // Initial transform for matching features. Optional. Always stores the
  // transform from current to previous frame.
  std::shared_ptr<Homography> initial_transform;

  ORBFeatureDescriptors orb;

  bool use_cv_tracking = false;

  FrameTrackingData(int width, int height, int extraction_levels,
                    bool _use_cv_tracking)
      : use_cv_tracking(_use_cv_tracking) {
    // Extraction pyramid.
    extraction_pyramid.clear();
    for (int i = 0, iwidth = width, iheight = height; i < extraction_levels;
         ++i) {
      extraction_pyramid.push_back(cv::Mat(iheight, iwidth, CV_8UC1));
      iwidth = (iwidth + 1) / 2;
      iheight = (iheight + 1) / 2;
    }
    CHECK_GE(extraction_levels, 1);
    // Frame is the same as first extraction level.
    frame = extraction_pyramid[0];

    if (!use_cv_tracking) {
      // Tracking pyramid for old c-interface.
      pyramid.resize(1);
      AllocatePyramid(width, height, &pyramid[0]);
    }
  }

  void BuildPyramid(int levels, int window_size, bool with_derivative) {
    if (use_cv_tracking) {
#if CV_MAJOR_VERSION == 3
      // No-op if not called for opencv 3.0 (c interface computes
      // pyramids in place).
      // OpenCV changed how window size gets specified from our radius setting
      // < 2.2 to diameter in 2.2+.
      cv::buildOpticalFlowPyramid(
          frame, pyramid, cv::Size(2 * window_size + 1, 2 * window_size + 1),
          levels, with_derivative);
      // Store max level for above pyramid.
      pyramid_levels = levels;
#endif
    }
  }

  void Reset(int frame_num_, int64 timestamp_) {
    frame_num = frame_num_;
    timestamp_usec = timestamp_;
    pyramid_levels = 0;
    ResetFeatures();
    neighborhoods.reset();
    orb.Reset();
  }

  void ResetFeatures() {
    features.clear();
    corner_responses.clear();
    octaves.clear();
    track_ids.clear();
    feature_source_map.clear();
    if (neighborhoods != nullptr) {
      neighborhoods->clear();
    }
    source = nullptr;
    features_initialized = false;
    last_feature_extraction_time = 0;
    num_original_extracted_and_tracked = -1;
    num_extracted_and_tracked = -1;
  }

  void PreAllocateFeatures(int num_features) {
    features.reserve(num_features);
    octaves.reserve(num_features);
    corner_responses.reserve(num_features);
    track_ids.reserve(num_features);
  }

  // Adds new feature and with required information.
  void AddFeature(const cv::Point2f& location, float corner_response,
                  int octave,
                  int track_id,                   // optional, -1 for none.
                  const cv::Mat* neighborhood) {  // optional.
    features.push_back(location);
    corner_responses.push_back(corner_response);
    octaves.push_back(octave);
    track_ids.push_back(track_id);
    if (neighborhoods != nullptr) {
      if (neighborhood) {
        neighborhoods->push_back(*neighborhood);
      } else {
        neighborhoods->push_back(cv::Mat());
      }
    }
  }

  void RemoveFeature(int pos) {
    DCHECK_LT(pos, features.size());
    features.erase(features.begin() + pos);
    feature_source_map.erase(feature_source_map.begin() + pos);
    corner_responses.erase(corner_responses.begin() + pos);
    octaves.erase(octaves.begin() + pos);
    track_ids.erase(track_ids.begin() + pos);
    if (neighborhoods) {
      neighborhoods->erase(neighborhoods->begin() + pos);
    }
  }

  // Stores grayscale square patch with length patch_size extracted at center in
  // image frame and stores result in patch.
  void ExtractPatch(const cv::Point2f& center, int patch_size, cv::Mat* patch) {
    CHECK(patch != nullptr);
    patch->create(patch_size, patch_size, CV_8UC1);
    cv::getRectSubPix(frame, cv::Size(patch_size, patch_size), center, *patch);
  }
};

// Data to be used across AddImage calls for long feature tracking.
struct RegionFlowComputation::LongTrackData {
  LongTrackData() = default;

  // Returns next id and records its start frame.
  int CreateNextTrackId(int start_frame, float motion_mag) {
    track_info[next_track_id] = TrackInfo(start_frame, motion_mag);
    const int result = next_track_id;

    // Advance.
    ++next_track_id;
    if (next_track_id < 0) {
      LOG(ERROR) << "Exhausted maximum possible ids. RegionFlowComputation "
                 << "instance lifetime is likely to be too long. Consider "
                 << "chunking the input.";
      next_track_id = 0;
    }

    return result;
  }

  // Returns last id that was created or -1 if an id was never created.
  int LastTrackId() const { return next_track_id - 1; }

  // Returns -1 if id is not present.
  int StartFrameForId(int id) const {
    auto id_iter = track_info.find(id);
    if (id_iter == track_info.end()) {
      return -1;
    } else {
      return id_iter->second.start_frame;
    }
  }

  // Clears buffered information for all features that are not present
  // anymore, i.e. not within the specified hash_set.
  void RemoveAbsentFeatureEntries(
      const absl::node_hash_set<int>& present_features) {
    auto entry = track_info.begin();
    while (entry != track_info.end()) {
      if (present_features.find(entry->first) == present_features.end()) {
        // Not present anymore, remove!
        auto to_erase = entry;
        ++entry;
        track_info.erase(to_erase);
      } else {
        ++entry;
      }
    }
  }

  float MotionMagForId(int id) const {
    auto id_iter = track_info.find(id);
    DCHECK(id_iter != track_info.end());
    return id_iter->second.motion_mag;
  }

  void UpdateMotion(int id, float motion_mag) {
    auto id_iter = track_info.find(id);
    DCHECK(id_iter != track_info.end());
    if (id_iter->second.motion_mag >= 0) {
      id_iter->second.motion_mag =
          id_iter->second.motion_mag * 0.5f + 0.5f * motion_mag;
    }
  }

  // Next id to be assigned to a new track.
  int next_track_id = 0;

  // Holds the previous result to seed the next frame.
  TrackedFeatureList prev_result;

  // Records for each track id some additional information.
  struct TrackInfo {
    TrackInfo() {}
    TrackInfo(int _start_frame, float _motion_mag)
        : start_frame(_start_frame), motion_mag(_motion_mag) {}
    int start_frame = 0;   // Start frame of track.
    float motion_mag = 0;  // Smoothed average motion. -1 for unknown.
  };

  std::unordered_map<int, TrackInfo> track_info;
};

template <class T>
std::unique_ptr<T> MakeUnique(T* ptr) {
  return std::unique_ptr<T>(ptr);
}

RegionFlowComputation::RegionFlowComputation(
    const RegionFlowComputationOptions& options, int frame_width,
    int frame_height)
    : options_(options),
      frame_width_(frame_width),
      frame_height_(frame_height) {
  switch (options_.gain_correct_mode()) {
    case RegionFlowComputationOptions::GAIN_CORRECT_DEFAULT_USER:
      // Do nothing, simply use supplied bounds.
      break;

    case RegionFlowComputationOptions::GAIN_CORRECT_VIDEO: {
      auto* gain_bias = options_.mutable_gain_bias_bounds();
      gain_bias->Clear();
      gain_bias->set_lower_gain(0.8f);
      gain_bias->set_upper_gain(1.2f);
      gain_bias->set_lower_bias(-0.2f);
      gain_bias->set_upper_bias(0.2f);
      gain_bias->set_min_inlier_weight(0.2f);
      gain_bias->set_min_inlier_fraction(0.6f);
      break;
    }

    case RegionFlowComputationOptions::GAIN_CORRECT_HDR: {
      auto* gain_bias = options_.mutable_gain_bias_bounds();
      gain_bias->Clear();
      gain_bias->set_lower_gain(0.8f);
      gain_bias->set_lower_gain(0.33f);
      gain_bias->set_upper_gain(3.0f);
      gain_bias->set_lower_bias(-0.5f);
      gain_bias->set_upper_bias(0.5f);
      gain_bias->set_min_inlier_weight(0.15f);
      gain_bias->set_min_inlier_fraction(0.6f);
      break;
    }

    case RegionFlowComputationOptions::GAIN_CORRECT_PHOTO_BURST: {
      auto* gain_bias = options_.mutable_gain_bias_bounds();
      gain_bias->Clear();
      gain_bias->set_min_inlier_fraction(0.6f);
      gain_bias->set_min_inlier_weight(0.1f);
      gain_bias->set_lower_gain(0.4f);
      gain_bias->set_upper_gain(2.5f);
      gain_bias->set_lower_bias(-0.6f);
      gain_bias->set_upper_bias(0.6f);
      break;
    }
  }

  CHECK_NE(options.tracking_options().output_flow_direction(),
           TrackingOptions::CONSECUTIVELY)
      << "Output direction must be either set to FORWARD or BACKWARD.";
  use_downsampling_ = options_.downsample_mode() !=
                      RegionFlowComputationOptions::DOWNSAMPLE_NONE;
  downsample_scale_ = 1;
  original_width_ = frame_width_;
  original_height_ = frame_height_;

  switch (options_.downsample_mode()) {
    case RegionFlowComputationOptions::DOWNSAMPLE_NONE:
      break;
    case RegionFlowComputationOptions::DOWNSAMPLE_TO_MAX_SIZE: {
      const float max_size = std::max(frame_width_, frame_height_);
      if (max_size > 1.03f * options_.downsampling_size()) {
        downsample_scale_ = max_size / options_.downsampling_size();
        if (options_.round_downsample_factor()) {
          downsample_scale_ = std::round(downsample_scale_);
        }
      }
      break;
    }
    case RegionFlowComputationOptions::DOWNSAMPLE_TO_MIN_SIZE: {
      const float min_size = std::min(frame_width_, frame_height_);
      if (min_size > 1.03f * options_.downsampling_size()) {
        downsample_scale_ = min_size / options_.downsampling_size();
        if (options_.round_downsample_factor()) {
          downsample_scale_ = std::round(downsample_scale_);
        }
      }
      break;
    }
    case RegionFlowComputationOptions::DOWNSAMPLE_BY_FACTOR:
    case RegionFlowComputationOptions::DOWNSAMPLE_TO_INPUT_SIZE: {
      CHECK_GE(options_.downsample_factor(), 1);
      downsample_scale_ = options_.downsample_factor();
      break;
    }
    case RegionFlowComputationOptions::DOWNSAMPLE_BY_SCHEDULE: {
      const int frame_area = frame_width_ * frame_height_;
      if (frame_area <= (16 * 1.03 / 9 * 360 * 360)) {
        downsample_scale_ =
            options_.downsample_schedule().downsample_factor_360p();
      } else if (frame_area <= (16 * 1.03 / 9 * 480 * 480)) {
        downsample_scale_ =
            options_.downsample_schedule().downsample_factor_480p();
      } else if (frame_area <= (16 * 1.03 / 9 * 720 * 720)) {
        downsample_scale_ =
            options_.downsample_schedule().downsample_factor_720p();
      } else {
        downsample_scale_ =
            options_.downsample_schedule().downsample_factor_1080p();
      }
      break;
    }
  }

  frame_width_ = std::round(frame_width_ / downsample_scale_);
  frame_height_ = std::round(frame_height_ / downsample_scale_);

  if (use_downsampling_ &&
      options_.downsample_mode() !=
          RegionFlowComputationOptions::DOWNSAMPLE_TO_INPUT_SIZE) {
    // Make downscaled size even.
    frame_width_ += frame_width_ % 2;
    frame_height_ += frame_height_ % 2;

    LOG(INFO) << "Using a downsampling scale of " << downsample_scale_;
  }

  // Make sure value is equal to local variable, in case someone uses that on
  // accident below.
  frame_width = frame_width_;
  frame_height = frame_height_;

  // Allocate temporary frames.
  switch (options_.image_format()) {
    case RegionFlowComputationOptions::FORMAT_RGB:
    case RegionFlowComputationOptions::FORMAT_BGR:
      curr_color_image_.reset(
          new cv::Mat(frame_height_, frame_width_, CV_8UC3));
      break;

    case RegionFlowComputationOptions::FORMAT_RGBA:
    case RegionFlowComputationOptions::FORMAT_BGRA:
      curr_color_image_.reset(
          new cv::Mat(frame_height_, frame_width_, CV_8UC4));
      break;

    case RegionFlowComputationOptions::FORMAT_GRAYSCALE:
      // Do nothing.
      break;
  }

  if (options_.compute_blur_score()) {
    corner_values_.reset(new cv::Mat(frame_height_, frame_width_, CV_32F));
    corner_filtered_.reset(new cv::Mat(frame_height_, frame_width_, CV_32F));
    corner_mask_.reset(new cv::Mat(frame_height_, frame_width_, CV_8U));
  }

  max_long_track_length_ = 1;
  switch (options_.tracking_options().tracking_policy()) {
    case TrackingOptions::POLICY_SINGLE_FRAME:
      if (options_.tracking_options().multi_frames_to_track() > 1) {
        LOG(ERROR) << "TrackingOptions::multi_frames_to_track is > 1, "
                   << "but tracking_policy is set to POLICY_SINGLE_FRAME. "
                   << "Consider using POLICY_MULTI_FRAME instead.";
      }

      frames_to_track_ = 1;
      break;
    case TrackingOptions::POLICY_MULTI_FRAME:
      CHECK_GT(options_.tracking_options().multi_frames_to_track(), 0);
      frames_to_track_ = options_.tracking_options().multi_frames_to_track();
      break;
    case TrackingOptions::POLICY_LONG_TRACKS:
      if (options_.tracking_options().multi_frames_to_track() > 1) {
        LOG(ERROR) << "TrackingOptions::multi_frames_to_track is > 1, "
                   << "but tracking_policy is set to POLICY_LONG_TRACKS. "
                   << "Use TrackingOptions::long_tracks_max_frames to set "
                   << "length of long feature tracks.";
      }

      if (options_.tracking_options().internal_tracking_direction() !=
          TrackingOptions::FORWARD) {
        LOG(ERROR) << "Long tracks are only supported if tracking direction "
                   << "is set to FORWARD. Adjusting direction to FORWARD. "
                   << "This does not affect the expected "
                   << "output_flow_direction";
        options_.mutable_tracking_options()->set_internal_tracking_direction(
            TrackingOptions::FORWARD);
      }

      frames_to_track_ = 1;
      max_long_track_length_ =
          options_.tracking_options().long_tracks_max_frames();
      long_track_data_.reset(new LongTrackData());
      break;
  }

  CHECK(!options_.gain_correction() || !IsVerifyLongFeatures())
      << "Gain correction mode with verification of long features is not "
      << "supported.";

  // Tracking algorithm dependent on cv support and flag.
  use_cv_tracking_ = options_.tracking_options().use_cv_tracking_algorithm();
#if CV_MAJOR_VERSION != 3
  if (use_cv_tracking_) {
    LOG(WARNING) << "Compiled without OpenCV 3.0 but cv_tracking_algorithm "
                 << "was requested. Falling back to older algorithm";
    use_cv_tracking_ = false;
  }
#endif

  if (options_.gain_correction()) {
    gain_image_.reset(new cv::Mat(frame_height_, frame_width_, CV_8UC1));
    if (!use_cv_tracking_) {
      gain_pyramid_.reset(new cv::Mat());
      AllocatePyramid(frame_width_, frame_height_, gain_pyramid_.get());
    }
  }

  // Determine number of levels at which to extract features. If lowest image
  // size for extraction is given, it overrides extraction levels.
  extraction_levels_ = options_.tracking_options().adaptive_extraction_levels();
  const int lowest_extraction_size =
      options_.tracking_options().adaptive_extraction_levels_lowest_size();
  if (lowest_extraction_size > 0) {
    const float frame_size = max(frame_width_, frame_height_);
    extraction_levels_ =
        1 + std::ceil(std::log2(frame_size / lowest_extraction_size) - 0.01);
  }
  extraction_levels_ = max(1, extraction_levels_);
  VLOG(1) << "Feature extraction will be done over " << extraction_levels_
          << " levels, starting at size (width, height): (" << frame_width_
          << ", " << frame_height_ << ")";

  feature_tmp_image_1_.reset(new cv::Mat(frame_height_, frame_width_, CV_32F));
  feature_tmp_image_2_.reset(new cv::Mat(frame_height_, frame_width_, CV_32F));

  // Allocate feature point arrays.
  max_features_ = options_.tracking_options().max_features();

  // Compute number of pyramid levels.
  float track_distance =
      hypot(frame_width_, frame_height_) *
      options_.tracking_options().fractional_tracking_distance();
  pyramid_levels_ = PyramidLevelsFromTrackDistance(track_distance);
  VLOG(1) << "Using pyramid levels: " << pyramid_levels_;

  // Compute settings for block based flow.
  const float block_size = options_.fast_estimation_block_size();
  CHECK_GT(block_size, 0) << "Need positive block size";

  block_width_ = block_size < 1 ? block_size * original_width_ : block_size;
  block_height_ = block_size < 1 ? block_size * original_height_ : block_size;
  // Ensure block_[width|height] is not zero.
  block_width_ = max(1, block_width_);
  block_height_ = max(1, block_height_);

  // Compute block pyramid levels.
  double min_block_dim = max(block_width_, block_height_);

  // Choose last_level such that
  // min_block_dim * 0.5^(last_level - 1) = min_block_size
  double last_level =
      (log(static_cast<double>(options_.fast_estimation_min_block_size())) -
       log(min_block_dim)) /
          log(0.5) +
      1;
  block_levels_ = max(2.0, floor(last_level));

  Reset();
}

RegionFlowComputation::~RegionFlowComputation() {}

bool RegionFlowComputation::AddImage(const cv::Mat& source,
                                     int64 timestamp_usec) {
  return AddImageAndTrack(source, cv::Mat(), timestamp_usec, Homography());
}

bool RegionFlowComputation::AddImageWithSeed(
    const cv::Mat& source, int64 timestamp_usec,
    const Homography& initial_transform) {
  return AddImageAndTrack(source, cv::Mat(), timestamp_usec, initial_transform);
}

bool RegionFlowComputation::AddImageWithMask(const cv::Mat& source,
                                             const cv::Mat& source_mask,
                                             int64 timestamp_usec) {
  return AddImageAndTrack(source, source_mask, timestamp_usec, Homography());
}

RegionFlowFeatureList* RegionFlowComputation::RetrieveRegionFlowFeatureList(
    bool compute_feature_descriptor, bool compute_match_descriptor,
    const cv::Mat* curr_color_image, const cv::Mat* prev_color_image) {
  return (RetrieveRegionFlowFeatureListImpl(
              0, compute_feature_descriptor, compute_match_descriptor,
              curr_color_image ? curr_color_image : nullptr,
              prev_color_image ? prev_color_image : nullptr)
              .release());
}

RegionFlowFrame* RegionFlowComputation::RetrieveRegionFlow() {
  return RetrieveMultiRegionFlow(0);
}

std::unique_ptr<RegionFlowFeatureList>
RegionFlowComputation::RetrieveRegionFlowFeatureListImpl(
    int track_index, bool compute_feature_descriptor,
    bool compute_match_descriptor, const cv::Mat* curr_color_image,
    const cv::Mat* prev_color_image) {
  CHECK_GT(region_flow_results_.size(), track_index);
  CHECK(region_flow_results_[track_index].get());

  std::unique_ptr<RegionFlowFeatureList> feature_list(
      std::move(region_flow_results_[track_index]));

  if (compute_feature_descriptor) {
    CHECK(curr_color_image != nullptr);
    CHECK_EQ(3, curr_color_image->channels());
    if (compute_match_descriptor) {
      CHECK(prev_color_image != nullptr);
      CHECK_EQ(3, prev_color_image->channels());
    }

    ComputeRegionFlowFeatureDescriptors(
        *curr_color_image,
        compute_match_descriptor ? prev_color_image : nullptr,
        options_.patch_descriptor_radius(), feature_list.get());
  } else {
    CHECK(!compute_match_descriptor) << "Set compute_feature_descriptor also "
                                     << "if setting compute_match_descriptor";
  }

  return feature_list;
}

RegionFlowFrame* RegionFlowComputation::RetrieveMultiRegionFlow(int frame) {
  std::unique_ptr<RegionFlowFeatureList> feature_list(
      RetrieveRegionFlowFeatureListImpl(frame,
                                        false,  // No descriptors.
                                        false,  // No match descriptors.
                                        nullptr, nullptr));

  std::unique_ptr<RegionFlowFrame> flow_frame(new RegionFlowFrame());

  RegionFlowFeatureListToRegionFlow(*feature_list, flow_frame.get());
  return flow_frame.release();
}

RegionFlowFeatureList*
RegionFlowComputation::RetrieveMultiRegionFlowFeatureList(
    int track_index, bool compute_feature_descriptor,
    bool compute_match_descriptor, const cv::Mat* curr_color_image,
    const cv::Mat* prev_color_image) {
  return (RetrieveRegionFlowFeatureListImpl(
              track_index, compute_feature_descriptor, compute_match_descriptor,
              curr_color_image ? curr_color_image : nullptr,
              prev_color_image ? prev_color_image : nullptr)
              .release());
}

bool RegionFlowComputation::InitFrame(const cv::Mat& source,
                                      const cv::Mat& source_mask,
                                      FrameTrackingData* data) {
  // Destination frame, CV_8U grayscale of dimension frame_width_ x
  // frame_height_.
  cv::Mat& dest_frame = data->frame;
  cv::Mat& dest_mask = data->mask;

  // Do we need to downsample image?
  const cv::Mat* source_ptr = &source;
  if (use_downsampling_ &&
      options_.downsample_mode() !=
          RegionFlowComputationOptions::DOWNSAMPLE_TO_INPUT_SIZE) {
    // Area based method best for downsampling.
    // For color images to temporary buffer.
    cv::Mat& resized = source.channels() == 1 ? dest_frame : *curr_color_image_;
    cv::resize(source, resized, resized.size(), 0, 0, CV_INTER_AREA);
    source_ptr = &resized;
    // Resize feature extraction mask if needed.
    if (!source_mask.empty()) {
      dest_mask.create(resized.rows, resized.cols, CV_8UC1);
      cv::resize(source_mask, dest_mask, dest_mask.size(), 0, 0, CV_INTER_NN);
    }
  } else if (!source_mask.empty()) {
    source_mask.copyTo(dest_mask);
  }

  // Stores as tiny frame before color conversion if requested.
  const auto& visual_options = options_.visual_consistency_options();
  if (visual_options.compute_consistency()) {
    // Allocate tiny image.
    const int type = source_ptr->type();
    const int dimension = visual_options.tiny_image_dimension();
    data->tiny_image.create(dimension, dimension, type);
    cv::resize(*source_ptr, data->tiny_image, data->tiny_image.size(), 0, 0,
               CV_INTER_AREA);
  }

  if (source_ptr->channels() == 1 &&
      options_.image_format() !=
          RegionFlowComputationOptions::FORMAT_GRAYSCALE) {
    options_.set_image_format(RegionFlowComputationOptions::FORMAT_GRAYSCALE);
    LOG(WARNING) << "#channels = 1, but image_format was not set to "
                    "FORMAT_GRAYSCALE. Assuming GRAYSCALE input.";
  }

  // Convert image to grayscale.
  switch (options_.image_format()) {
    case RegionFlowComputationOptions::FORMAT_RGB:
      if (3 != source_ptr->channels()) {
        LOG(ERROR) << "Expecting 3 channel input for RGB.";
        return false;
      }
      cv::cvtColor(*source_ptr, dest_frame, cv::COLOR_RGB2GRAY);
      break;

    case RegionFlowComputationOptions::FORMAT_BGR:
      if (3 != source_ptr->channels()) {
        LOG(ERROR) << "Expecting 3 channel input for BGR.";
        return false;
      }
      cv::cvtColor(*source_ptr, dest_frame, cv::COLOR_BGR2GRAY);
      break;

    case RegionFlowComputationOptions::FORMAT_RGBA:
      if (4 != source_ptr->channels()) {
        LOG(ERROR) << "Expecting 4 channel input for RGBA.";
        return false;
      }
      cv::cvtColor(*source_ptr, dest_frame, cv::COLOR_RGBA2GRAY);
      break;

    case RegionFlowComputationOptions::FORMAT_BGRA:
      if (4 != source_ptr->channels()) {
        LOG(ERROR) << "Expecting 4 channel input for BGRA.";
        return false;
      }
      cv::cvtColor(*source_ptr, dest_frame, cv::COLOR_BGRA2GRAY);
      break;

    case RegionFlowComputationOptions::FORMAT_GRAYSCALE:
      if (1 != source_ptr->channels()) {
        LOG(ERROR) << "Expecting 1 channel input for GRAYSCALE.";
        return false;
      }
      CHECK_EQ(1, source_ptr->channels());
      if (source_ptr != &dest_frame) {
        source_ptr->copyTo(dest_frame);
      }
      break;
  }

  // Do histogram equalization.
  if (options_.histogram_equalization()) {
    cv::equalizeHist(dest_frame, dest_frame);
  }

  // Compute mean for gain correction.
  if (options_.gain_correction()) {
    data->mean_intensity = cv::mean(dest_frame)[0];
  }

  // Consistency checks; not input governed.
  CHECK_EQ(dest_frame.cols, frame_width_);
  CHECK_EQ(dest_frame.rows, frame_height_);

  data->BuildPyramid(pyramid_levels_,
                     options_.tracking_options().tracking_window_size(),
                     options_.compute_derivative_in_pyramid());

  return true;
}

bool RegionFlowComputation::AddImageAndTrack(
    const cv::Mat& source, const cv::Mat& source_mask, int64 timestamp_usec,
    const Homography& initial_transform) {
  VLOG(1) << "Processing frame " << frame_num_ << " at " << timestamp_usec;
  MEASURE_TIME << "AddImageAndTrack";

  if (options_.downsample_mode() ==
      RegionFlowComputationOptions::DOWNSAMPLE_TO_INPUT_SIZE) {
    if (frame_width_ != source.cols || frame_height_ != source.rows) {
      LOG(ERROR) << "Source input dimensions incompatible with "
                 << "DOWNSAMPLE_TO_INPUT_SIZE. frame_width_: " << frame_width_
                 << ", source.cols: " << source.cols
                 << ", frame_height_: " << frame_height_
                 << ", source.rows: " << source.rows;
      return false;
    }

    if (!source_mask.empty()) {
      if (frame_width_ != source_mask.cols ||
          frame_height_ != source_mask.rows) {
        LOG(ERROR) << "Input mask dimensions incompatible with "
                   << "DOWNSAMPLE_TO_INPUT_SIZE";
        return false;
      }
    }
  } else {
    if (original_width_ != source.cols || original_height_ != source.rows) {
      LOG(ERROR) << "Source input dimensions differ from those specified "
                 << "in the constructor";
      return false;
    }
    if (!source_mask.empty()) {
      if (original_width_ != source_mask.cols ||
          original_height_ != source_mask.rows) {
        LOG(ERROR) << "Input mask dimensions incompatible with those "
                   << "specified in the constructor";
        return false;
      }
    }
  }

  // Create data queue element for current frame. If queue is full, reuse the
  // data field from front in a circular buffer, otherwise add a new one.
  if (data_queue_.size() > frames_to_track_) {
    data_queue_.push_back(std::move(data_queue_.front()));
    data_queue_.pop_front();
  } else {
    data_queue_.push_back(MakeUnique(new FrameTrackingData(
        frame_width_, frame_height_, extraction_levels_, use_cv_tracking_)));
  }

  FrameTrackingData* curr_data = data_queue_.back().get();
  curr_data->Reset(frame_num_, timestamp_usec);

  if (!IsModelIdentity(initial_transform)) {
    CHECK_EQ(1, frames_to_track_) << "Initial transform is not supported "
                                  << "for multi frame tracking";
    Homography transform = initial_transform;
    if (downsample_scale_ != 1) {
      const float scale = 1.0f / downsample_scale_;
      transform = CoordinateTransform(initial_transform, scale);
    }
    curr_data->initial_transform.reset(new Homography(transform));
  }

  if (!InitFrame(source, source_mask, curr_data)) {
    LOG(ERROR) << "Could not init frame.";
    return false;
  }

  // Precompute blur score from original (not pre-blurred) frame.
  cv::Mat& curr_frame = curr_data->frame;
  curr_blur_score_ =
      options_.compute_blur_score() ? ComputeBlurScore(curr_frame) : -1;

  if (options_.pre_blur_sigma() > 0) {
    cv::GaussianBlur(curr_frame, curr_frame, cv::Size(0, 0),
                     options_.pre_blur_sigma(), options_.pre_blur_sigma());
  }

  // By default, create empty region flows for as many frames as we want to
  // track.
  region_flow_results_.clear();
  for (int i = 0; i < frames_to_track_; ++i) {
    region_flow_results_.push_back(MakeUnique(new RegionFlowFeatureList()));
    InitializeRegionFlowFeatureList(region_flow_results_.back().get());
  }

  // Do we have enough frames to start tracking?
  const bool synthetic_tracks =
      options_.use_synthetic_zero_motion_tracks_all_frames() ||
      (frame_num_ == 0 &&
       options_.use_synthetic_zero_motion_tracks_first_frame());

  int curr_frames_to_track = frames_to_track_;
  // Update frames-to-track to match actual frames that we can track.
  if (!synthetic_tracks) {
    curr_frames_to_track = min(frame_num_, frames_to_track_);
  }

  // Compute region flows for all frames being tracked.
  const auto internal_flow_direction =
      options_.tracking_options().internal_tracking_direction();
  const bool invert_flow = internal_flow_direction !=
                           options_.tracking_options().output_flow_direction();

  switch (internal_flow_direction) {
    case TrackingOptions::FORWARD:
      if (long_track_data_ != nullptr && curr_frames_to_track > 0) {
        // Long feature tracking.
        TrackedFeatureList curr_result;
        ComputeRegionFlow(-1, 0, synthetic_tracks, invert_flow,
                          &long_track_data_->prev_result, &curr_result,
                          region_flow_results_[0].get());
        long_track_data_->prev_result.swap(curr_result);
      } else {
        // Track from the closest frame last, so that the last set of features
        // updated in FrameTrackingData are from the closest one.
        for (int i = curr_frames_to_track; i >= 1; --i) {
          ComputeRegionFlow(-i, 0, synthetic_tracks, invert_flow, nullptr,
                            nullptr, region_flow_results_[i - 1].get());
        }
      }
      break;
    case TrackingOptions::BACKWARD:
      for (int i = 1; i <= curr_frames_to_track; ++i) {
        if (!synthetic_tracks && i > 1) {
          InitializeFeatureLocationsFromPreviousResult(-i + 1, -i);
        }
        ComputeRegionFlow(0, -i, synthetic_tracks, invert_flow, nullptr,
                          nullptr, region_flow_results_[i - 1].get());
      }
      break;
    case TrackingOptions::CONSECUTIVELY:
      const bool invert_flow_forward =
          TrackingOptions::FORWARD !=
          options_.tracking_options().output_flow_direction();
      const bool invert_flow_backward = !invert_flow_forward;
      for (int i = curr_frames_to_track; i >= 1; --i) {
        // Compute forward flow.
        ComputeRegionFlow(-i, 0, synthetic_tracks, invert_flow_forward, nullptr,
                          nullptr, region_flow_results_[i - 1].get());
        if (region_flow_results_[i - 1]->unstable()) {
          // If forward flow is unstable, compute backward flow.
          ComputeRegionFlow(0, -i, synthetic_tracks, invert_flow_backward,
                            nullptr, nullptr,
                            region_flow_results_[i - 1].get());
        }
      }
      break;
  }

  if (frames_to_track_ == 1) {
    const int num_features = region_flow_results_.front()->feature_size();
    if (frame_num_ == 0) {
      curr_num_features_avg_ = num_features;
    } else {
      // Low pass filter number of current features.
      constexpr float kAlpha = 0.3f;
      curr_num_features_avg_ =
          (1.0f - kAlpha) * curr_num_features_avg_ + kAlpha * num_features;
    }
  }

  ++frame_num_;

  return true;
}

cv::Mat RegionFlowComputation::GetGrayscaleFrameFromResults() {
  CHECK_GT(data_queue_.size(), 0) << "Empty queue, was AddImage* called?";
  FrameTrackingData* curr_data = data_queue_.back().get();
  CHECK(curr_data);
  return curr_data->frame;
}

void RegionFlowComputation::GetFeatureTrackInliers(
    bool skip_estimation, TrackedFeatureList* features,
    TrackedFeatureView* inliers) const {
  CHECK(features != nullptr);
  CHECK(inliers != nullptr);
  inliers->clear();
  if (skip_estimation) {
    inliers->reserve(features->size());
    for (auto& feature : *features) {
      inliers->push_back(&feature);
    }
  } else {
    ComputeBlockBasedFlow(features, inliers);
  }
}

float RegionFlowComputation::ComputeVisualConsistency(
    FrameTrackingData* previous, FrameTrackingData* current) const {
  CHECK_EQ(previous->frame_num + 1, current->frame_num);
  const int total = previous->tiny_image.total();
  CHECK_GT(total, 0) << "Tiny image dimension set to zero.";
  current->tiny_image_diff =
      FrameDifferenceMedian(previous->tiny_image, current->tiny_image) *
      (1.0f / total);

  return fabs(previous->tiny_image_diff - current->tiny_image_diff);
}

void RegionFlowComputation::ComputeRegionFlow(
    int from, int to, bool synthetic_tracks, bool invert_flow,
    const TrackedFeatureList* prev_result, TrackedFeatureList* curr_result,
    RegionFlowFeatureList* feature_list) {
  MEASURE_TIME << "Compute RegionFlow.";
  // feature_tracks should be in the outer scope since the inliers form a view
  // on them (store pointers to features stored in feature_tracks).
  TrackedFeatureList feature_tracks;
  TrackedFeatureView feature_inliers;

  FrameTrackingData* data1 = nullptr;
  FrameTrackingData* data2 = nullptr;
  float frac_long_features_rejected = 0;
  float visual_consistency = 0;

  if (synthetic_tracks) {
    const float step =
        options_.tracking_options().synthetic_zero_motion_grid_step();
    ZeroMotionGridTracks(original_width_, original_height_, step, step,
                         &feature_tracks);
    GetFeatureTrackInliers(true /* skip_estimation */, &feature_tracks,
                           &feature_inliers);
  } else {
    const int index1 = data_queue_.size() + from - 1;
    const int index2 = data_queue_.size() + to - 1;
    CHECK_GE(index1, 0);
    CHECK_LT(index1, data_queue_.size());
    CHECK_GE(index2, 0);
    CHECK_LT(index2, data_queue_.size());
    data1 = data_queue_[index1].get();
    data2 = data_queue_[index2].get();

    std::unique_ptr<Homography> initial_transform;
    if (index1 + 1 == index2) {
      // Forward track, check if initial transform present.
      if (data2->initial_transform != nullptr) {
        initial_transform = absl::make_unique<Homography>(
            ModelInvert(*data2->initial_transform));
      }
    } else if (index1 - 1 == index2) {
      // Backward track, check if initial transform present.
      if (data1->initial_transform != nullptr) {
        initial_transform.reset(new Homography(*data1->initial_transform));
      }
    }

    if (std::abs(from - to) == 1 &&
        options_.visual_consistency_options().compute_consistency()) {
      FrameTrackingData* later_data = data1;
      FrameTrackingData* earlier_data = data2;
      if (from < to) {
        std::swap(later_data, earlier_data);
      }

      visual_consistency = ComputeVisualConsistency(earlier_data, later_data);
    }

    bool track_features = true;
    bool force_feature_extraction_next_frame = false;
    if (options_.tracking_options().wide_baseline_matching()) {
      CHECK(initial_transform == nullptr)
          << "Can't use wide baseline matching and initial transform as the "
          << "same time.";

      WideBaselineMatchFeatures(data1, data2, &feature_tracks);
      track_features =
          options_.tracking_options().refine_wide_baseline_matches();
      if (track_features) {
        initial_transform = absl::make_unique<Homography>(
            HomographyAdapter::Embed(AffineModelFromFeatures(&feature_tracks)));
        feature_tracks.clear();
      } else {
        GetFeatureTrackInliers(options_.no_estimation_mode(), &feature_tracks,
                               &feature_inliers);
      }
    }

    if (track_features) {
      ExtractFeatures(prev_result, data1);

      if (initial_transform != nullptr) {
        InitializeFeatureLocationsFromTransform(from, to, *initial_transform);
      }

      // Compute tracks with gain correction if requested.
      bool gain_correction = options_.gain_correction();
      const float triggering_ratio =
          options_.gain_correction_triggering_ratio();
      if (options_.gain_correction() && triggering_ratio > 0) {
        // Only compute gain if change in intensity across frames
        // is sufficiently large.
        const float intensity_ratio =
            std::max(data1->mean_intensity, data2->mean_intensity) /
            (std::min(data1->mean_intensity, data2->mean_intensity) + 1e-6f);
        gain_correction = intensity_ratio > triggering_ratio;
      }

      const bool gain_hypotheses =
          options_.gain_correction_multiple_hypotheses();

      // Trigger feature extraction on next frame every time we perform
      // gain correction.
      force_feature_extraction_next_frame = gain_correction;

      // Backup FrameTrackingData if needed for reset when using multiple
      // hypothesis.
      std::unique_ptr<FrameTrackingData> wo_gain_data2;
      if (gain_correction && gain_hypotheses) {
        wo_gain_data2.reset(new FrameTrackingData(*data2));
      }

      TrackFeatures(data1, data2, &gain_correction,
                    &frac_long_features_rejected, &feature_tracks);
      GetFeatureTrackInliers(options_.no_estimation_mode(), &feature_tracks,
                             &feature_inliers);

      // Second pass: If gain correction was successful and multiple hypotheses,
      // are requested run again without it.
      if (gain_correction && gain_hypotheses) {
        // Re-run without gain correction.
        TrackedFeatureList wo_gain_tracks;
        TrackedFeatureView wo_gain_inliers;

        gain_correction = false;
        TrackFeatures(data1, wo_gain_data2.get(), &gain_correction, nullptr,
                      &wo_gain_tracks);
        GetFeatureTrackInliers(options_.no_estimation_mode(), &wo_gain_tracks,
                               &wo_gain_inliers);

        // Only use gain correction if it is better than tracking
        // without gain correction, i.e gain correction should result in more
        // inliers, at least by the specified fractional improvement.
        const float improvement_weight =
            1.0f + options_.gain_correction_inlier_improvement_frac();
        const int gain_count = feature_inliers.size();
        const int wo_gain_count = wo_gain_inliers.size();
        if (gain_count < wo_gain_count * improvement_weight) {
          // Reject gain result, insufficient improvement. Use result without
          // gain correction instead.
          feature_tracks.swap(wo_gain_tracks);
          feature_inliers.swap(wo_gain_inliers);
          std::swap(*data2, *wo_gain_data2);
          VLOG(1) << "Rejecting gain correction. Number of inliers with "
                  << "gain: " << gain_count
                  << ", without gain: " << wo_gain_count;
          force_feature_extraction_next_frame = false;
        }
      }
    }  // end if track features.

    if (data1->num_original_extracted_and_tracked < 0) {
      // Record initial track.
      data1->num_original_extracted_and_tracked = feature_inliers.size();
    }

    if (force_feature_extraction_next_frame) {
      data2->num_extracted_and_tracked = 0;
    } else {
      data2->num_extracted_and_tracked = feature_inliers.size();
    }

    // Forward initial number of tracked features.
    data2->num_original_extracted_and_tracked =
        data1->num_original_extracted_and_tracked;
  }

  // Convert tracks to region flow.
  if (invert_flow) {
    InvertFeatureList(feature_tracks, &feature_tracks);
  }

  const float flow_magnitude = TrackedFeatureViewToRegionFlowFeatureList(
      feature_inliers, curr_result, feature_list);

  // Assign unique ids to the features.
  for (auto& feature : *feature_list->mutable_feature()) {
    feature.set_feature_id(++feature_count_);
  }

  if (from != to) {
    // Record average flow magnitude (normalized w.r.t. tracking distance in
    // frames).
    flow_magnitudes_.push_back(flow_magnitude / std::abs(from - to));
    const int kMaxMagnitudeRecords = 10;
    // Limit to only most recent observations.
    while (flow_magnitudes_.size() > kMaxMagnitudeRecords) {
      flow_magnitudes_.pop_front();
    }

    // Adaptively size pyramid based on previous observations.
    // 130% of previous maximum.
    if (options_.tracking_options().adaptive_tracking_distance() &&
        flow_magnitudes_.size() > 2) {
      pyramid_levels_ = PyramidLevelsFromTrackDistance(
          *std::max_element(flow_magnitudes_.begin(), flow_magnitudes_.end()) *
          1.3f);
    }
  }

  // Check if sufficient features found, set corresponding flags.
  if (!HasSufficientFeatures(*feature_list)) {
    feature_list->set_unstable(true);
    // If region flow is unstable, then the tracked features in the "to" frame
    // should not be relied upon for reuse.
    if (data2 != nullptr) {
      data2->ResetFeatures();
    }
  }

  // Store additional information in feature_list.
  feature_list->set_frac_long_features_rejected(frac_long_features_rejected);
  feature_list->set_visual_consistency(visual_consistency);
  if (invert_flow) {
    if (data2 != nullptr) {
      feature_list->set_timestamp_usec(data2->timestamp_usec);
    }
  } else {
    if (data1 != nullptr) {
      feature_list->set_timestamp_usec(data1->timestamp_usec);
    }
  }
  if (data1 != nullptr) {
    *feature_list->mutable_actively_discarded_tracked_ids() = {
        data1->actively_discarded_tracked_ids.begin(),
        data1->actively_discarded_tracked_ids.end()};
    data1->actively_discarded_tracked_ids.clear();
  }

  feature_list->set_match_frame((to - from) * (invert_flow ? -1 : 1));
}

// Resets computation by setting frame_num_ == 0. No need to clear other data
// structures, since previous frames are used only once frame_num_ > 0.
void RegionFlowComputation::Reset() {
  frame_num_ = 0;
  data_queue_.clear();
  flow_magnitudes_.clear();
}

namespace {

struct FloatPointerComparator {
  bool operator()(const float* lhs, const float* rhs) const {
    return *lhs > *rhs;
  }
};

// Invoker for ParallelFor. Needs to be copyable.
// Extracts features from a 2nd moment gradient response image (eig_image)
// by grid-based thresholding (removing feature responses below
// local_quality_level * maximum cell value or lowest_quality_level) and
// non-maxima suppression via dilation. Results are output in corner_pointers
// and partially sorted (limited to max_cell_features, highest first).
class GridFeatureLocator {
 public:
  GridFeatureLocator(int frame_width, int frame_height, int block_width,
                     int block_height, int bins_per_row,
                     float local_quality_level, float lowest_quality_level,
                     int max_cell_features,
                     std::vector<std::vector<const float*>>* corner_pointers,
                     cv::Mat* eig_image, cv::Mat* tmp_image)
      : frame_width_(frame_width),
        frame_height_(frame_height),
        block_width_(block_width),
        block_height_(block_height),
        bins_per_row_(bins_per_row),
        local_quality_level_(local_quality_level),
        lowest_quality_level_(lowest_quality_level),
        max_cell_features_(max_cell_features),
        corner_pointers_(corner_pointers),
        eig_image_(eig_image),
        tmp_image_(tmp_image) {}

  void operator()(const BlockedRange2D& range) const {
    // Iterate over views.
    for (int bin_y = range.rows().begin(); bin_y != range.rows().end();
         ++bin_y) {
      for (int bin_x = range.cols().begin(); bin_x != range.cols().end();
           ++bin_x) {
        const int view_x = bin_x * block_width_;
        const int view_y = bin_y * block_height_;
        const int view_end_x = min(frame_width_, (bin_x + 1) * block_width_);
        const int view_end_y = min(frame_height_, (bin_y + 1) * block_height_);

        // Guarantee at least one pixel in each dimension.
        if (view_x >= view_end_x || view_y >= view_end_y) {
          continue;
        }

        cv::Mat eig_view(*eig_image_, cv::Range(view_y, view_end_y),
                         cv::Range(view_x, view_end_x));
        cv::Mat tmp_view(*tmp_image_, cv::Range(view_y, view_end_y),
                         cv::Range(view_x, view_end_x));

        // Ignore features below quality level.
        double maximum = 0;
        cv::minMaxLoc(eig_view, nullptr, &maximum, nullptr, nullptr);
        double lowest_quality = std::max<double>(maximum * local_quality_level_,
                                                 lowest_quality_level_);

        // Copy borders that do not get dilated below.
        cv::Rect borders[4] = {
            cv::Rect(0, 0, eig_view.cols, 1),                  // top
            cv::Rect(0, 0, 1, eig_view.rows),                  // left
            cv::Rect(0, eig_view.rows - 1, eig_view.cols, 1),  // bottom
            cv::Rect(eig_view.cols - 1, 0, 1, eig_view.rows)};

        for (int k = 0; k < 4; ++k) {
          cv::Mat dst_view(tmp_view, borders[k]);
          cv::Mat(eig_view, borders[k]).copyTo(dst_view);
        }

        // Non-maxima suppression.
        if (tmp_view.rows > 2 && tmp_view.cols > 2) {
          // Remove border from dilate, as cv::dilate reads out of
          // bound values otherwise.
          cv::Mat dilate_src =
              cv::Mat(eig_view, cv::Range(1, eig_view.rows - 1),
                      cv::Range(1, eig_view.cols - 1));

          cv::Mat dilate_dst =
              cv::Mat(tmp_view, cv::Range(1, tmp_view.rows - 1),
                      cv::Range(1, tmp_view.cols - 1));
          cv::Mat kernel(3, 3, CV_32F);
          kernel.setTo(1.0);
          cv::dilate(dilate_src, dilate_dst, kernel);
        }

        const int grid_pos = bin_y * bins_per_row_ + bin_x;
        std::vector<const float*>& grid_cell = (*corner_pointers_)[grid_pos];

        // Iterate over view in image domain as we store feature location
        // pointers w.r.t. original frame.
        for (int i = view_y; i < view_end_y; ++i) {
          const float* tmp_ptr = tmp_image_->ptr<float>(i);
          const float* eig_ptr = eig_image_->ptr<float>(i);
          for (int j = view_x; j < view_end_x; ++j) {
            const float max_supp_value = tmp_ptr[j];
            if (max_supp_value > lowest_quality &&
                max_supp_value == eig_ptr[j]) {
              // This is a local maxima -> store in list.
              grid_cell.push_back(eig_ptr + j);
            }
          }
        }

        const int level_max_elems =
            min<int>(max_cell_features_, grid_cell.size());
        std::partial_sort(grid_cell.begin(),
                          grid_cell.begin() + level_max_elems, grid_cell.end(),
                          FloatPointerComparator());
      }
    }
  }

 private:
  int frame_width_;
  int frame_height_;
  int block_width_;
  int block_height_;
  int bins_per_row_;
  float local_quality_level_;
  float lowest_quality_level_;
  int max_cell_features_;
  std::vector<std::vector<const float*>>* corner_pointers_;
  cv::Mat* eig_image_;
  cv::Mat* tmp_image_;
};

// Sets (2 * N + 1) x (2 * N + 1) neighborhood of the passed mask to K
// or adds K to the existing mask if add is set to true.
template <int N, int K, bool add>
inline void SetMaskNeighborhood(int mask_x, int mask_y, cv::Mat* mask) {
  DCHECK_EQ(mask->type(), CV_8U);
  const int mask_start_x = max(0, mask_x - N);
  const int mask_end_x = min(mask->cols - 1, mask_x + N);
  const int mask_dx = mask_end_x - mask_start_x + 1;
  const int mask_start_y = max(0, mask_y - N);
  const int mask_end_y = min(mask->rows - 1, mask_y + N);
  DCHECK_LE(mask_start_x, mask_end_x);
  DCHECK_LE(mask_start_y, mask_end_y);

  if (!add) {
    for (int i = mask_start_y; i <= mask_end_y; ++i) {
      uint8* mask_ptr = mask->ptr<uint8>(i) + mask_start_x;
      memset(mask_ptr, K, mask_dx * sizeof(*mask_ptr));
    }
  } else {
    for (int i = mask_start_y; i <= mask_end_y; ++i) {
      uint8* mask_ptr = mask->ptr<uint8>(i);
      for (int j = mask_start_x; j <= mask_end_x; ++j) {
        mask_ptr[j] = (mask_ptr[j] & 0x7F) + K;  // Limit to 128.
      }
    }
  }
}

}  // namespace.

void RegionFlowComputation::AdaptiveGoodFeaturesToTrack(
    const std::vector<cv::Mat>& extraction_pyramid, int max_features,
    float mask_scale, cv::Mat* mask, FrameTrackingData* data) {
  CHECK(data != nullptr);
  CHECK(feature_tmp_image_1_.get() != nullptr);
  CHECK(feature_tmp_image_2_.get() != nullptr);

  cv::Mat* eig_image = feature_tmp_image_1_.get();
  cv::Mat* tmp_image = feature_tmp_image_2_.get();

  const auto& tracking_options = options_.tracking_options();

  // Setup grid information.
  const float block_size = tracking_options.adaptive_features_block_size();
  CHECK_GT(block_size, 0) << "Need positive block size";

  int block_width = block_size < 1 ? block_size * frame_width_ : block_size;
  int block_height = block_size < 1 ? block_size * frame_height_ : block_size;

  // Ensure valid block width and height regardless of settings.
  block_width = max(1, block_width);
  block_height = max(1, block_height);

  bool use_harris = tracking_options.corner_extraction_method() ==
                    TrackingOptions::EXTRACTION_HARRIS;

  const int adaptive_levels =
      options_.tracking_options().adaptive_features_levels();

  // For Harris negative values are possible, so lowest_quality level
  // is being ignored.
  const float lowest_quality_level =
      use_harris ? -100.0f
                 : tracking_options.min_eig_val_settings()
                       .adaptive_lowest_quality_level();

  const float local_quality_level =
      use_harris
          ? tracking_options.harris_settings().feature_quality_level()
          : tracking_options.min_eig_val_settings().feature_quality_level();

  bool use_fast = tracking_options.corner_extraction_method() ==
                  TrackingOptions::EXTRACTION_FAST;
  cv::Ptr<cv::FastFeatureDetector> fast_detector;
  if (use_fast) {
    fast_detector = cv::FastFeatureDetector::create(
        tracking_options.fast_settings().threshold());
  }

  // Extract features at multiple scales and adaptive block sizes.
  for (int e = 0, step = 1; e < extraction_pyramid.size(); ++e) {
    if (data->features.size() >= max_features) {
      break;
    }

    const cv::Mat& image = extraction_pyramid[e];
    const int rows = image.rows;
    const int cols = image.cols;

    // Compute corner response.
    constexpr int kBlockSize = 3;
    constexpr double kHarrisK = 0.04;  // Harris magical constant as
                                       // set by OpenCV.
    std::vector<cv::KeyPoint> fast_keypoints;
    if (e == 0) {
      MEASURE_TIME << "Corner extraction";
      CHECK_EQ(rows, frame_height_);
      CHECK_EQ(cols, frame_width_);

      if (use_fast) {
        fast_detector->detect(image, fast_keypoints);
      } else if (use_harris) {
        cv::cornerHarris(image, *eig_image, kBlockSize, kBlockSize, kHarrisK);
      } else {
        cv::cornerMinEigenVal(image, *eig_image, kBlockSize);
      }
    } else {
      // Compute corner response on a down-scaled image and upsample.
      step *= 2;
      CHECK_EQ(rows, (extraction_pyramid[e - 1].rows + 1) / 2);
      CHECK_EQ(cols, (extraction_pyramid[e - 1].cols + 1) / 2);

      if (use_fast) {
        fast_detector->detect(image, fast_keypoints);

        for (int j = 0; j < fast_keypoints.size(); ++j) {
          fast_keypoints[j].pt.x *= step;
          fast_keypoints[j].pt.y *= step;
        }
      } else {
        // Use tmp_image to compute eigen-values on resized images.
        cv::Mat eig_view(*tmp_image, cv::Range(0, rows), cv::Range(0, cols));

        if (use_harris) {
          cv::cornerHarris(image, eig_view, kBlockSize, kBlockSize, kHarrisK);
        } else {
          cv::cornerMinEigenVal(image, eig_view, kBlockSize);
        }

        // Upsample (without interpolation) eig_view to match frame size.
        eig_image->setTo(0);
        for (int r = 0, r_up = 0; r < rows && r_up < frame_height_;
             ++r, r_up += step) {
          const float* ptr = eig_view.ptr<float>(r);
          float* up_ptr = eig_image->ptr<float>(r_up);
          for (int c = 0, c_up = 0; c < cols && c_up < frame_width_;
               ++c, c_up += step) {
            up_ptr[c_up] = ptr[c];
          }
        }
      }  // end if use_fast
    }    // end if e.

    if (use_fast) {
      // TODO: Perform grid based feature detection.
      std::sort(fast_keypoints.begin(), fast_keypoints.end(),
                [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return b.response < a.response;
                });

      for (int j = 0; j < fast_keypoints.size(); ++j) {
        const int corner_y = fast_keypoints[j].pt.y;
        const int corner_x = fast_keypoints[j].pt.x;
        const int mask_x = corner_x * mask_scale;
        const int mask_y = corner_y * mask_scale;

        // Test if neighboring element is already set.
        if (mask->at<uint8>(mask_y, mask_x) >= 1) {
          continue;
        }

        SetMaskNeighborhood<2, 1, false>(mask_x, mask_y, mask);

        // `e` stands for which pyramid layer this feature is extracted from.
        data->AddFeature(cv::Point2f(corner_x, corner_y),
                         min(1.0f, fast_keypoints[j].response), e,
                         -1,  // No track id assigned yet. Only assign ids
                              // to successfully tracked features.
                         nullptr);
      }
    } else {
      // Iterate over adaptive pyramid levels.
      int level_width = block_width;
      int level_height = block_height;
      for (int level = 0; level < adaptive_levels; ++level) {
        // Perform grid based threshold and non-maxima suppression.
        const int bins_per_column =
            std::ceil(static_cast<float>(frame_height_) / level_height);
        const int bins_per_row =
            std::ceil(static_cast<float>(frame_width_) / level_width);
        const int num_bins = bins_per_row * bins_per_column;
        const int level_max_features = max_features - data->features.size();
        if (level_max_features < 0) {
          break;
        }

        std::vector<std::vector<const float*>> corner_pointers(num_bins);
        for (int k = 0; k < num_bins; ++k) {
          corner_pointers[k].reserve(level_max_features);
        }

        GridFeatureLocator locator(
            frame_width_, frame_height_, level_width, level_height,
            bins_per_row, local_quality_level, lowest_quality_level,
            level_max_features, &corner_pointers, eig_image, tmp_image);

        ParallelFor2D(0, bins_per_column, 0, bins_per_row, 1, locator);

        // Round robin across bins, add one feature per bin, until
        // max_features is hit.
        bool more_features_available = true;

        // Index of next to be processed corner in corner_points[k] array.
        std::vector<int> corner_index(num_bins, 0);
        while (more_features_available &&
               data->features.size() < max_features) {
          more_features_available = false;

          // Add one feature per bin (stratified sampling).
          for (int k = 0; k < num_bins; ++k) {
            if (corner_index[k] >= corner_pointers[k].size()) {
              continue;
            }

            const float* corner_ptr = corner_pointers[k][corner_index[k]];
            // Advance.
            ++corner_index[k];
            if (corner_index[k] < corner_pointers[k].size() - 1) {
              more_features_available = true;
            }

            // Map corner pointer to x and y location.
            const int offset = reinterpret_cast<const uint8*>(corner_ptr) -
                               eig_image->ptr<const uint8>(0);

            const int corner_y = offset / eig_image->step[0];
            const int corner_x =
                (offset - eig_image->step[0] * corner_y) / sizeof(*corner_ptr);

            // Ensure corner is at least 2 pixel away from boundary.
            if (corner_x < 2 || corner_x > frame_width_ - 2 || corner_y < 2 ||
                corner_y > frame_height_ - 2) {
              continue;
            }

            const int mask_x = corner_x * mask_scale;
            const int mask_y = corner_y * mask_scale;

            // Test if neighboring element is already set.
            if (mask->at<uint8>(mask_y, mask_x) >= 1) {
              continue;
            }

            SetMaskNeighborhood<2, 1, false>(mask_x, mask_y, mask);

            // `e` stands for which pyramid layer the feature is extracted from.
            data->AddFeature(
                cv::Point2f(corner_x, corner_y),
                min(1.0f, *corner_ptr * options_.corner_response_scale()), e,
                -1,  // No track id assigned yet. Only assign ids
                     // to successfully tracked features.
                nullptr);
          }  // end bins.
        }    // end while.

        if (level + 1 < adaptive_levels) {
          level_width = (level_width + 1) / 2;
          level_height = (level_height + 1) / 2;
        }
      }  // end adaptive level.
    }    // end use_fast
  }      // end extraction level.

  // If adaptive_levels or extraction_levels > 1, for 2nd or larger level, we
  // can potentially add corners above the max_features threshold. In this case
  // truncation performs well without having to resort to sorting the features,
  // as 2nd or larger level add features of lower corner response that pass the
  // locality test (as the neighborhood size decreases).
  if (data->features.size() > max_features) {
    data->features.resize(max_features);
    data->corner_responses.resize(max_features);
    data->octaves.resize(max_features);
    data->track_ids.resize(max_features);
  }
}

AffineModel RegionFlowComputation::AffineModelFromFeatures(
    TrackedFeatureList* features) const {
  CHECK(features != nullptr);

  // Downscaled domain as output.
  MotionEstimation motion_estimation(MotionEstimationOptions(), frame_width_,
                                     frame_height_);

  RegionFlowFrame region_flow;
  region_flow.set_frame_width(original_width_);
  region_flow.set_frame_height(original_height_);

  TrackedFeatureView feature_view;
  ComputeBlockBasedFlow(features, &feature_view);

  RegionFlowFeatureList feature_list;
  TrackedFeatureViewToRegionFlowFeatureList(feature_view, nullptr,
                                            &feature_list);

  return FitAffineModel(feature_list);
}

void RegionFlowComputation::ZeroMotionGridFeatures(
    int frame_width, int frame_height, float frac_grid_step_x,
    float frac_grid_step_y, RegionFlowFeatureList* result) {
  CHECK(result != nullptr);
  result->Clear();

  TrackedFeatureList features;
  const int border_dist = ZeroMotionGridTracks(
      frame_width, frame_height, frac_grid_step_x, frac_grid_step_y, &features);

  result->set_frame_width(frame_width);
  result->set_frame_height(frame_height);
  result->set_distance_from_border(border_dist);

  for (const auto& feature : features) {
    RegionFlowFeature* new_feature = result->add_feature();
    new_feature->set_x(feature.point.x());
    new_feature->set_y(feature.point.y());
    new_feature->set_dx(feature.flow.x());
    new_feature->set_dy(feature.flow.y());
  }
}

void RegionFlowComputation::DenseZeroMotionSamples(
    int frame_width, int frame_height, float frac_diameter, float frac_steps_x,
    float frac_steps_y, RegionFlowFeatureList* result) {
  CHECK(result != nullptr);

  // Ensure patch fits into frame.
  const int radius =
      max<int>(1, min<int>(min<int>(frame_width / 2 - 1, frame_height / 2 - 1),
                           hypot(frame_width, frame_height) * frac_diameter) /
                      2);
  result->Clear();
  result->set_frame_width(frame_width);
  result->set_frame_height(frame_height);
  result->set_distance_from_border(radius);

  const int start = radius;
  const int end_y = frame_height - radius;
  const int end_x = frame_width - radius;

  const int steps_x = max<int>(1, frame_width * frac_steps_x);
  const int steps_y = max<int>(1, frame_height * frac_steps_y);
  for (int y = start; y < end_y; y += steps_y) {
    for (int x = start; x < end_x; x += steps_x) {
      RegionFlowFeature* new_feature = result->add_feature();
      new_feature->set_x(x);
      new_feature->set_y(y);
      new_feature->set_dx(0);
      new_feature->set_dy(0);
    }
  }
}

namespace {

bool PointOutOfBound(const Vector2_f& point, int frame_width,
                     int frame_height) {
  if (point.x() < 0 || point.y() < 0 || point.x() > frame_width - 1 ||
      point.y() > frame_height - 1) {
    return true;
  }
  return false;
}

}  // namespace.

int RegionFlowComputation::ZeroMotionGridTracks(int frame_width,
                                                int frame_height,
                                                float frac_grid_step_x,
                                                float frac_grid_step_y,
                                                TrackedFeatureList* results) {
  CHECK(results);
  auto& tracked_features = *results;
  tracked_features.clear();

  const int grid_step_x =
      max(1, static_cast<int>(frac_grid_step_x * frame_width));
  const int grid_step_y =
      max(1, static_cast<int>(frac_grid_step_y * frame_height));

  const int num_features_x = (frame_width - 1) / grid_step_x;
  const int num_features_y = (frame_height - 1) / grid_step_y;
  const int max_features = num_features_x * num_features_y;

  // Only track features in one frame for synthetic features.
  tracked_features.reserve(max_features);
  const int border_dist_x = grid_step_x / 2;
  const int border_dist_y = grid_step_y / 2;
  for (int i = 0, y = border_dist_y; i < num_features_y;
       ++i, y += grid_step_y) {
    for (int j = 0, x = border_dist_x; j < num_features_x;
         ++j, x += grid_step_x) {
      TrackedFeature tracked_feature(Vector2_f(x, y), Vector2_f(0, 0), 0.0f,
                                     0.0f,
                                     -1);  // No track id assigned.
      tracked_features.push_back(tracked_feature);
    }
  }

  return min(border_dist_x, border_dist_y);
}

bool RegionFlowComputation::GainCorrectFrame(const cv::Mat& reference_frame,
                                             const cv::Mat& input_frame,
                                             float reference_mean,
                                             float input_mean,
                                             cv::Mat* calibrated_frame) const {
  CHECK(calibrated_frame);
  CHECK_EQ(reference_frame.rows, input_frame.rows);
  CHECK_EQ(reference_frame.cols, input_frame.cols);

  // Do not attempt gain correction for tiny images.
  if (std::min(reference_frame.rows, reference_frame.cols) < 10) {
    VLOG(1) << "Tiny image, aborting gain correction.";
    return false;
  }

  GainBiasModel gain_bias;
  if (options_.fast_gain_correction()) {
    const int kMinMean = 5;
    if (input_mean < kMinMean) {
      return false;  // Badly exposed.
    }
    const float gain = reference_mean / input_mean;
    if (gain < options_.gain_bias_bounds().lower_gain() ||
        gain > options_.gain_bias_bounds().upper_gain()) {
      return false;  // Unstable: Out of bound.
    }

    gain_bias.set_gain_c1(gain);
  }

  constexpr float kMaxFastGain = 1.12f;
  if (!options_.fast_gain_correction() || gain_bias.gain_c1() > kMaxFastGain) {
    // Estimate tone change w.r.t. reference_frame.
    RegionFlowFeatureList zero_features;
    DenseZeroMotionSamples(
        frame_width_, frame_height_, options_.frac_gain_feature_size(),
        options_.frac_gain_step(), options_.frac_gain_step(), &zero_features);

    ClipMask<1> reference_mask;
    ClipMask<1> input_mask;
    ToneEstimation::ComputeClipMask(ClipMaskOptions(), reference_frame,
                                    &reference_mask);

    ToneEstimation::ComputeClipMask(ClipMaskOptions(), input_frame,
                                    &input_mask);

    ColorToneMatches tone_matches;
    ToneMatchOptions tone_match_options;
    tone_match_options.set_patch_radius(zero_features.distance_from_border() -
                                        1);

    // Nothing to extract.
    if (tone_match_options.patch_radius() < 1) {
      VLOG(1) << "Patch radius is < 1, aborting gain correction.";
      return false;
    }

    ToneEstimation::ComputeToneMatches<1>(
        tone_match_options, zero_features, input_frame, reference_frame,
        input_mask, reference_mask, &tone_matches);

    // Only attempt estimation if not too much frame area is clipped.
    if (tone_matches[0].size() <= 0.5 * zero_features.feature_size()) {
      VLOG(1) << "Too much frame area is clipped for gain correction.";
      return false;
    }

    ToneEstimation::EstimateGainBiasModel(5,  // number of irls iterations.
                                          &tone_matches, &gain_bias);

    if (!ToneEstimation::IsStableGainBiasModel(
            options_.gain_bias_bounds(), gain_bias, tone_matches, nullptr)) {
      VLOG(1) << "Unstable gain-bias model.";
      return false;
    }
  }

  GainBiasModelMethods::MapImageIndependent<1>(gain_bias,
                                               false,  // log_domain.
                                               true,   // normalized_model.
                                               input_frame, calibrated_frame);
  return true;
}

void RegionFlowComputation::WideBaselineMatchFeatures(
    FrameTrackingData* from_data_ptr, FrameTrackingData* to_data_ptr,
    TrackedFeatureList* results) {
#if (defined(__ANDROID__) || defined(__APPLE__) || defined(__EMSCRIPTEN__)) && \
    !defined(CV_WRAPPER_3X)
  LOG(FATAL) << "Supported on only with OpenCV 3.0. "
             << "Use bazel build flag : --define CV_WRAPPER=3X";
#else  // (defined(__ANDROID__) || defined(__APPLE__) ||
       // defined(__EMSCRIPTEN__)) && !defined(CV_WRAPPER_3X)
  results->clear();

  const auto& frame1 = from_data_ptr->frame;
  auto& data1 = from_data_ptr->orb;

  const auto& frame2 = to_data_ptr->frame;
  auto& data2 = to_data_ptr->orb;

  cv::Ptr<cv::ORB> orb = cv::ORB::create(max_features_);

  // Compute ORB features in frame1.
  if (!data1.computed) {
    orb->detect(frame1, data1.key_points);
    orb->compute(frame1, data1.key_points, data1.descriptors);
    data1.computed = true;
  }

  const int num_features = data1.key_points.size();
  if (num_features == 0) {
    // No features found, probably black or extremely blurry frame. Return empty
    // results.
    VLOG(1) << "Couldn't extract any features. Frame probably empty.";
    return;
  }

  // Compute ORB features in frame2.
  if (!data2.computed) {
    orb->detect(frame2, data2.key_points);
    orb->compute(frame2, data2.key_points, data2.descriptors);
    data2.computed = true;
  }

  // Match feature descriptors.
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> matches;
  matcher.knnMatch(data2.descriptors,  // Query.
                   data1.descriptors,  // Train.
                   matches,
                   2);  // 2 closest matches per descriptor.

  results->reserve(matches.size());

  // Get successfully matched features.
  for (const auto& match : matches) {
    if (match.size() > 1 &&
        match[0].distance < options_.tracking_options().ratio_test_threshold() *
                                match[1].distance) {
      // Passed ratio test.
      const cv::Point2f& feature_location =
          data1.key_points[match[0].trainIdx].pt;
      const cv::Point2f& match_location =
          data2.key_points[match[0].queryIdx].pt;

      const Vector2_f feature_point(feature_location.x, feature_location.y);
      const Vector2_f flow =
          Vector2_f(match_location.x, match_location.y) - feature_point;

      TrackedFeature tracked_feature(feature_point * downsample_scale_,
                                     flow * downsample_scale_,
                                     match[0].distance, 0.0f,
                                     -1);  // no track id assigned.

      if (PointOutOfBound(tracked_feature.point, original_width_,
                          original_height_)) {
        continue;
      }

      VLOG(2) << "Flow: " << tracked_feature.flow << " @ "
              << tracked_feature.point;
      results->push_back(tracked_feature);
    }
  }
#endif  // (defined(__ANDROID__) || defined(__APPLE__) ||
        // defined(__EMSCRIPTEN__)) && !defined(CV_WRAPPER_3X)
}

void RegionFlowComputation::RemoveAbsentFeatures(
    const TrackedFeatureList& prev_result, FrameTrackingData* data) {
  CHECK(long_track_data_ != nullptr);

  // Build hash set of track ids.
  absl::node_hash_set<int> track_ids;
  for (const auto& feature : prev_result) {
    DCHECK_NE(feature.track_id, -1);
    track_ids.insert(feature.track_id);
  }

  long_track_data_->RemoveAbsentFeatureEntries(track_ids);

  // Remove indices (backwards to not destroy index positions).
  for (int k = data->track_ids.size() - 1; k >= 0; --k) {
    if (track_ids.find(data->track_ids[k]) == track_ids.end()) {
      data->RemoveFeature(k);
    }
  }
}

void RegionFlowComputation::RemoveFeaturesOutsideMask(FrameTrackingData* data) {
  if (data->mask.empty()) {
    return;
  }

  // Remove indices (backwards to not destroy index positions).
  for (int k = data->features.size() - 1; k >= 0; --k) {
    const int x = static_cast<int>(data->features[k].x + 0.5);
    const int y = static_cast<int>(data->features[k].y + 0.5);
    if (data->mask.at<uint8>(y, x) == 0) {
      data->RemoveFeature(k);
    }
  }
}

void RegionFlowComputation::ExtractFeatures(
    const TrackedFeatureList* prev_result, FrameTrackingData* data) {
  MEASURE_TIME << "ExtractFeatures";
  if (!options_.tracking_options().adaptive_good_features_to_track()) {
    LOG(FATAL) << "Deprecated! Activate adaptive_good_features_to_track "
               << "in TrackingOptions";
  }

  // Check if features can simply be re-used.
  if (!data->features.empty()) {
    if (prev_result) {
      // Long feature tracking case, remove features,
      // whose ids are not present anymore (these are mainly outliers
      // that got removed during DetermineRegionFlowInliers call).
      RemoveAbsentFeatures(*prev_result, data);
    }

    if (data->last_feature_extraction_time == 0) {
      // Features already extracted from this frame.
      CHECK_EQ(data->corner_responses.size(), data->features.size());
      CHECK_EQ(data->octaves.size(), data->features.size());
      VLOG(1) << "Features already present (extracted from this frame)";
      return;
    }

    // Remove features that lie outside feature extraction mask.
    RemoveFeaturesOutsideMask(data);

    CHECK_EQ(data->corner_responses.size(), data->features.size());
    CHECK_EQ(data->octaves.size(), data->features.size());

    float feature_fraction = 0;
    if (data->num_original_extracted_and_tracked > 0) {
      feature_fraction = data->num_extracted_and_tracked * 1.0f /
                         data->num_original_extracted_and_tracked;
    }

    // If features were not tracked from too far away, reuse them unless the
    // number of tracked features is below a threshold percentage of the
    // original source features.
    const int max_frame_distance =
        options_.tracking_options().reuse_features_max_frame_distance();
    const float min_survived_frac =
        options_.tracking_options().reuse_features_min_survived_frac();

    if (data->last_feature_extraction_time <= max_frame_distance &&
        feature_fraction > min_survived_frac) {
      VLOG(1) << "Features already present, (tracked "
              << data->last_feature_extraction_time << " times)";
      return;
    }
  }
  // If execution reaches this point, new features will be extracted.

  // Scale feature_distance by sqrt(scale). Sqrt is purely a heuristic choice.
  float min_feature_distance =
      options_.tracking_options().min_feature_distance();
  if (min_feature_distance < 1) {
    min_feature_distance *= hypot(frame_width_, frame_height_);
  }
  if (options_.tracking_options().distance_downscale_sqrt()) {
    min_feature_distance =
        std::round(min_feature_distance / std::sqrt(downsample_scale_));
  }

  // Result mask that ensures we don't place features too closely.
  const float mask_dim = max(1.0f, min_feature_distance * 0.5f);
  const float mask_scale = 1.0f / mask_dim;
  cv::Mat mask = cv::Mat::zeros(std::ceil(frame_height_ * mask_scale),
                                std::ceil(frame_width_ * mask_scale), CV_8U);

  // Initialize mask from frame's feature extraction mask, by downsampling and
  // negating the latter mask.
  if (!data->mask.empty()) {
    cv::resize(data->mask, mask, mask.size(), 0, 0, CV_INTER_NN);
    for (int y = 0; y < mask.rows; ++y) {
      uint8* mask_ptr = mask.ptr<uint8>(y);
      for (int x = 0; x < mask.cols; ++x) {
        mask_ptr[x] = mask_ptr[x] == 0 ? 1 : 0;
      }
    }
  }

  data->ResetFeatures();
  const int features_to_allocate =
      prev_result ? prev_result->size() * 1.2f : max_features_ / 2;
  data->PreAllocateFeatures(features_to_allocate);

  if (IsVerifyLongFeatures()) {
    if (data->neighborhoods == nullptr) {
      data->neighborhoods.reset(new std::vector<cv::Mat>());
    }
    data->neighborhoods->reserve(features_to_allocate);
  }

  CHECK_EQ(data->extraction_pyramid.size(), extraction_levels_);
  for (int i = 1; i < extraction_levels_; ++i) {
    // Need factor 2 as OpenCV stores image + gradient pairs when
    // "with_derivative" is set to true.
    const int layer_stored_in_pyramid =
        options_.compute_derivative_in_pyramid() ? 2 * i : i;
    const bool index_within_limit =
        (layer_stored_in_pyramid < data->pyramid.size());
    if (index_within_limit && options_.compute_derivative_in_pyramid() &&
        i <= data->pyramid_levels) {
      // Just re-use from already computed pyramid.
      data->extraction_pyramid[i] = data->pyramid[layer_stored_in_pyramid];
    } else {
      cv::pyrDown(data->extraction_pyramid[i - 1], data->extraction_pyramid[i],
                  data->extraction_pyramid[i].size());
    }
  }

  if (prev_result) {
    // Seed feature mask and results with tracking ids.
    CHECK(long_track_data_ != nullptr);
    const int max_track_length =
        options_.tracking_options().long_tracks_max_frames();
    // Drop a feature with a propability X, such that all qualifying
    // features are dropped with a 95% probability within the interval
    // [.8, 1.2] * long_tracks_max_frames.
    const int lower_max_track_length = max(1.0f, 0.8f * max_track_length);
    const int upper_max_track_length = 1.2f * max_track_length;

    // Ensure interval is positive.
    const int interval_length =
        upper_max_track_length - lower_max_track_length + 1;
    // Drop probability: p  == > survival: 1 - p
    // (1 - p) ^ interval_length >= 5%    // only 5% survival chance across
    //                                    // all frames
    //  ==> p <= 1 - (0.05 ^ (1.0 / interval_length)
    // Ensure positive probability.
    const float drop_permil =
        max(1.0, (1.0 - pow(0.05, 1.0 / interval_length)));

    unsigned int seed = 900913;  // = Google in leet :)
    std::default_random_engine rand_gen(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    // Mask out locations.
    // For FORWARD output flow, we need to add flow to obtain the match
    // position, for BACKWARD output flow, flow is inverted, so that feature
    // locations already point to locations in the current frame.
    CHECK_EQ(options_.tracking_options().internal_tracking_direction(),
             TrackingOptions::FORWARD);
    float match_sign = options_.tracking_options().output_flow_direction() ==
                               TrackingOptions::FORWARD
                           ? 1.0f
                           : 0.0f;
    const float inv_downsample_scale_ = 1.0f / downsample_scale_;

    for (const auto& feature : *prev_result) {
      // Need to convert to downsampled domain.
      const Vector2_f pos =
          (feature.point + feature.flow * match_sign) * inv_downsample_scale_;

      const int track_id = feature.track_id;
      if (track_id < 0) {
        // TODO: Use LOG_FIRST_N here.
        LOG_IF(WARNING,
               []() {
                 static int k = 0;
                 return k++ < 2;
               }())
            << "Expecting an assigned track id, "
            << "skipping feature.";
        continue;
      }

      // Skip features for which the track would get too long.
      const int start_frame = long_track_data_->StartFrameForId(track_id);
      if (start_frame < 0) {
        LOG(ERROR) << "Id is not present, skipping feature.";
        continue;
      }

      if (data->frame_num - start_frame >= lower_max_track_length &&
          distribution(rand_gen) <= drop_permil) {
        data->actively_discarded_tracked_ids.push_back(track_id);
        continue;
      }

      const int mask_x = pos.x() * mask_scale;
      const int mask_y = pos.y() * mask_scale;

      // Skip if already occupied by too many features. This allows paths
      // to "join", without having to explicitly represent this.
      // Value of 2 improves number of connected features.
      constexpr int kMaxFeaturesPerBin = 1;
      if (mask.at<uint8>(mask_y, mask_x) >= kMaxFeaturesPerBin) {
        data->actively_discarded_tracked_ids.push_back(track_id);
        continue;
      }

      SetMaskNeighborhood<2, 1, false>(mask_x, mask_y, &mask);

      // Copy results to features.
      const cv::Mat* neighborhood = (options_.verify_long_features() &&
                                     feature.orig_neighborhood != nullptr)
                                        ? feature.orig_neighborhood.get()
                                        : nullptr;
      data->AddFeature(cv::Point2f(pos.x(), pos.y()), feature.corner_response,
                       feature.octave, feature.track_id, neighborhood);
    }
  }

  // Extracts additional features in regions excluding the mask and adds them to
  // data.
  AdaptiveGoodFeaturesToTrack(data->extraction_pyramid, max_features_,
                              mask_scale, &mask, data);

  const int num_features = data->features.size();
  CHECK_EQ(num_features, data->octaves.size());
  CHECK_EQ(num_features, data->corner_responses.size());
  CHECK_EQ(num_features, data->track_ids.size());
}

// Selects features based on lambda evaluator: bool (int index)
// Performs inplace moves and final resize operation.
template <class Eval>
int RegionFlowComputation::InplaceFeatureSelection(
    FrameTrackingData* data, std::vector<std::vector<int>*> int_vecs,
    std::vector<std::vector<float>*> float_vecs, const Eval& eval) {
  int num_selected_features = 0;
  const int num_features = data->features.size();
  DCHECK_EQ(num_features, data->corner_responses.size());
  DCHECK_EQ(num_features, data->octaves.size());
  DCHECK_EQ(num_features, data->track_ids.size());
  DCHECK_EQ(num_features, data->feature_source_map.size());
  if (data->neighborhoods != nullptr) {
    DCHECK_EQ(num_features, data->neighborhoods->size());
  }

  for (const auto vec_ptr : int_vecs) {
    DCHECK_EQ(num_features, vec_ptr->size());
  }
  for (const auto vec_ptr : float_vecs) {
    DCHECK_EQ(num_features, vec_ptr->size());
  }

  for (int i = 0; i < num_features; ++i) {
    DCHECK_LE(num_selected_features, i);
    if (eval(i)) {
      data->features[num_selected_features] = data->features[i];
      data->corner_responses[num_selected_features] = data->corner_responses[i];
      data->octaves[num_selected_features] = data->octaves[i];
      data->track_ids[num_selected_features] = data->track_ids[i];
      data->feature_source_map[num_selected_features] =
          data->feature_source_map[i];
      if (data->neighborhoods != nullptr) {
        (*data->neighborhoods)[num_selected_features] =
            (*data->neighborhoods)[i];
      }

      for (auto* vec_ptr : int_vecs) {
        (*vec_ptr)[num_selected_features] = (*vec_ptr)[i];
      }
      for (auto* vec_ptr : float_vecs) {
        (*vec_ptr)[num_selected_features] = (*vec_ptr)[i];
      }
      ++num_selected_features;
    }
  }

  // Trim to number of selected features.
  data->features.resize(num_selected_features);
  data->corner_responses.resize(num_selected_features);
  data->octaves.resize(num_selected_features);
  data->track_ids.resize(num_selected_features);
  data->feature_source_map.resize(num_selected_features);
  if (data->neighborhoods != nullptr) {
    data->neighborhoods->resize(num_selected_features);
  }

  for (const auto vec_ptr : int_vecs) {
    vec_ptr->resize(num_selected_features);
  }
  for (const auto vec_ptr : float_vecs) {
    vec_ptr->resize(num_selected_features);
  }

  return num_selected_features;
}

void RegionFlowComputation::TrackFeatures(FrameTrackingData* from_data_ptr,
                                          FrameTrackingData* to_data_ptr,
                                          bool* gain_correction_ptr,
                                          float* frac_long_features_rejected,
                                          TrackedFeatureList* results_ptr) {
  MEASURE_TIME << "TrackFeatures";

  FrameTrackingData& data1 = *from_data_ptr;
  const cv::Mat& frame1 = data1.frame;
  const std::vector<cv::Point2f>& features1 = data1.features;
  const std::vector<float>& corner_responses1 = data1.corner_responses;
  const std::vector<int>& octaves1 = data1.octaves;
  // New features will be assigned new track ids.
  std::vector<int>& track_ids1 = data1.track_ids;

  FrameTrackingData& data2 = *to_data_ptr;
  cv::Mat& frame2 = data2.frame;
  std::vector<cv::Point2f>& features2 = data2.features;
  std::vector<float>& corner_responses2 = data2.corner_responses;
  std::vector<int>& octaves2 = data2.octaves;
  std::vector<int>& track_ids2 = data2.track_ids;

  bool& gain_correction = *gain_correction_ptr;
  TrackedFeatureList& results = *results_ptr;
  std::vector<int>& feature_source_map = data2.feature_source_map;

  // Start frame for new features. Minimum of from and to.
  const int min_frame =
      std::min(from_data_ptr->frame_num, to_data_ptr->frame_num);

  feature_source_map.clear();
  results.clear();

  const int num_features = features1.size();
  if (num_features == 0) {
    // No features found, probably black or extremely blurry frame.
    // Return empty results.
    VLOG(1) << "Couldn't find any features to track. Frame probably empty.";
    return;
  }

  int tracking_flags = 0;
  // Check if features in the destination are already initialized. If so, use
  // their location as initial guess.
  if (!data2.features_initialized) {
    data2.ResetFeatures();
    features2.resize(num_features);
    corner_responses2.resize(num_features);
    octaves2.resize(num_features);
    data2.source = from_data_ptr;
  } else {
    CHECK_EQ(data2.source, from_data_ptr);
    CHECK_EQ(num_features, features2.size());
    tracking_flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
  }

  const int track_win_size = options_.tracking_options().tracking_window_size();
  CHECK_GT(track_win_size, 1) << "Needs to be at least 2 pixels in each "
                              << "direction";

  // Proceed with gain correction only if it succeeds, and set flag accordingly.
  bool frame1_gain_reference = true;
  if (gain_correction) {
    cv::Mat reference_frame = frame1;
    cv::Mat input_frame = frame2;
    float reference_mean = data1.mean_intensity;
    float input_mean = data2.mean_intensity;
    // Use brighter frame as reference, if requested.
    if (options_.gain_correction_bright_reference() &&
        data1.mean_intensity < data2.mean_intensity) {
      std::swap(input_frame, reference_frame);
      std::swap(input_mean, reference_mean);
      frame1_gain_reference = false;
    }

    // Gain correct and update output with success result.
    gain_correction =
        GainCorrectFrame(reference_frame, input_frame, reference_mean,
                         input_mean, gain_image_.get());
  }

#if CV_MAJOR_VERSION == 3
  // OpenCV changed how window size gets specified from our radius setting
  // < 2.2 to diameter in 2.2+.
  const cv::Size cv_window_size(track_win_size * 2 + 1, track_win_size * 2 + 1);

  cv::TermCriteria cv_criteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
      options_.tracking_options().tracking_iterations(), 0.02f);

  cv::_InputArray input_frame1(data1.pyramid);
  cv::_InputArray input_frame2(data2.pyramid);
#endif

  // Using old c-interface for OpenCV's 2.2 tracker.
  CvTermCriteria criteria;
  criteria.type = CV_TERMCRIT_EPS | CV_TERMCRIT_ITER;
  criteria.max_iter = options_.tracking_options().tracking_iterations();
  criteria.epsilon = 0.02f;

  feature_track_error_.resize(num_features);
  feature_status_.resize(num_features);
  if (use_cv_tracking_) {
#if CV_MAJOR_VERSION == 3
    if (gain_correction) {
      if (!frame1_gain_reference) {
        input_frame1 = cv::_InputArray(*gain_image_);
      } else {
        input_frame2 = cv::_InputArray(*gain_image_);
      }
    }

    if (options_.tracking_options().klt_tracker_implementation() ==
        TrackingOptions::KLT_OPENCV) {
      cv::calcOpticalFlowPyrLK(input_frame1, input_frame2, features1, features2,
                               feature_status_, feature_track_error_,
                               cv_window_size, pyramid_levels_, cv_criteria,
                               tracking_flags);
    } else {
      LOG(ERROR) << "Tracking method unspecified.";
      return;
    }
#endif
  } else {
    LOG(ERROR) << "only cv tracking is supported.";
    return;
  }

  // Inherit corner response and octaves from extracted features.
  corner_responses2 = corner_responses1;
  octaves2 = octaves1;

  // Remember mapping from destination to source index;
  // Fill feature_source_map with 0 ... num_features - 1.
  feature_source_map.resize(num_features);
  std::iota(feature_source_map.begin(), feature_source_map.end(), 0);

  // Init track ids.
  track_ids2 = std::vector<int>(num_features, -1);  // Unassigned by default.

  // Select features that were successfully tracked from data1 to data2.
  int num_valid_features = InplaceFeatureSelection(
      to_data_ptr, {&feature_source_map}, {&feature_track_error_},
      [this](int i) -> bool { return feature_status_[i] == 1; });

  // Init neighborhoods if needed.
  if (IsVerifyLongFeatures()) {
    // data1 should be initialized at this point.
    CHECK(data1.neighborhoods != nullptr);
    if (data2.neighborhoods == nullptr) {
      data2.neighborhoods.reset(new std::vector<cv::Mat>());
      data2.neighborhoods->resize(num_valid_features);
    }
  }

  // Remember last id threshold before assigning new ones.
  const int prev_id_threshold =
      long_track_data_ != nullptr ? long_track_data_->LastTrackId() : 0;

  std::vector<int> ids_to_verify;
  std::vector<int> motions_to_verify;
  if (long_track_data_) {
    // Compute motion for each feature and average magnitude.
    std::vector<float> motion_mag(num_valid_features, 0);
    float avg_motion_mag = 0;

    for (int i = 0; i < num_valid_features; ++i) {
      const int match_idx = feature_source_map[i];
      const cv::Point2f diff =
          (features2[i] - features1[match_idx]) * downsample_scale_;
      const float norm = std::abs(diff.x) + std::abs(diff.y);
      motion_mag[i] = norm;
      avg_motion_mag += norm;
    }

    if (num_valid_features > 0) {
      avg_motion_mag /= num_valid_features;
    }

    bool is_duplicated = num_valid_features > 0 && avg_motion_mag < kZeroMotion;

    const float max_acc = options_.max_long_feature_acceleration();

    // Minimum motion for stable ratio test.
    constexpr float kMinMotion = 1.0f;

    // Initialize all track_ids of data2.
    int num_restarted_tracks = 0;
    for (int i = 0; i < num_valid_features; ++i) {
      const int match_idx = feature_source_map[i];
      if (track_ids1[match_idx] < 0) {
        const float motion_mag_arg = is_duplicated ? -1 : motion_mag[i];
        track_ids1[match_idx] =
            long_track_data_->CreateNextTrackId(min_frame, motion_mag_arg);
        track_ids2[i] = track_ids1[match_idx];
      } else if (!is_duplicated) {
        // Check for change in acceleration for non duplicated frame.
        const float prev_motion_mag =
            long_track_data_->MotionMagForId(track_ids1[match_idx]);

        // Test for acceleration or deacceleration, but only if previous
        // motion was known (from non duplicated frame).
        if (prev_motion_mag >= 0 &&
            (motion_mag[i] > max_acc * std::max(kMinMotion, prev_motion_mag) ||
             prev_motion_mag > max_acc * std::max(kMinMotion, motion_mag[i]))) {
          // Start new track or flag for testing.
          if (options_.verify_long_feature_acceleration()) {
            // Keep track id and update motion. If feature is removed and
            // therefore motion was updated incorrectly, motion wont be used
            // again.
            track_ids2[i] = track_ids1[match_idx];
            long_track_data_->UpdateMotion(track_ids1[match_idx],
                                           motion_mag[i]);
            ids_to_verify.push_back(i);
            motions_to_verify.push_back(motion_mag[i]);
          } else {
            ++num_restarted_tracks;
            track_ids2[i] =
                long_track_data_->CreateNextTrackId(min_frame, motion_mag[i]);
          }
        } else {
          long_track_data_->UpdateMotion(track_ids1[match_idx], motion_mag[i]);
          track_ids2[i] = track_ids1[match_idx];
        }
      } else {
        // Duplicated frame with existing track, re-use id without updating
        // motion.
        track_ids2[i] = track_ids1[match_idx];
      }

      if (IsVerifyLongFeatures()) {
        cv::Mat& mat1 = (*data1.neighborhoods)[match_idx];
        cv::Mat& mat2 = (*data2.neighborhoods)[i];
        if (mat1.empty()) {
          data1.ExtractPatch(features1[match_idx], track_win_size, &mat1);
        }
        data2.ExtractPatch(features2[i], track_win_size, &mat2);
      }
    }
    VLOG(1) << "Restarted tracks: " << num_restarted_tracks;
  }  // end long track data.

  if (!ids_to_verify.empty() &&
      ids_to_verify.size() <
          options_.verify_long_feature_trigger_ratio() * num_valid_features) {
    // Reset feature ids, instead of triggering verification.
    VLOG(1) << "Canceling feature verification, resetting tracks: "
            << ids_to_verify.size() << " of " << num_valid_features;
    for (int k = 0; k < ids_to_verify.size(); ++k) {
      const int id = ids_to_verify[k];
      track_ids2[id] =
          long_track_data_->CreateNextTrackId(min_frame, motions_to_verify[k]);
    }
    ids_to_verify.clear();
    motions_to_verify.clear();
  }

  // Distance between source location x and tracked-back f^(-1)(y) location
  // starting at the tracked location y = f(x): x - f^(-1)(f(x)).
  // Close to zero for tracks that could be verified.
  std::vector<float> verify_distance(num_valid_features, 0);

  // Compile list of indices we need to verify via backtracking.
  std::vector<int> feat_ids_to_verify;
  if (options_.verify_features()) {
    feat_ids_to_verify.resize(num_valid_features);
    std::iota(feat_ids_to_verify.begin(), feat_ids_to_verify.end(), 0);
  } else if (options_.verify_long_feature_acceleration()) {
    feat_ids_to_verify = ids_to_verify;
  }

  VLOG(1) << "Verifying: " << feat_ids_to_verify.size() << " out of "
          << num_valid_features;
  if (!feat_ids_to_verify.empty()) {
    const int num_to_verify = feat_ids_to_verify.size();
    std::vector<cv::Point2f> verify_features;
    std::vector<cv::Point2f> verify_features_tracked;
    verify_features.reserve(num_to_verify);
    verify_features_tracked.reserve(num_to_verify);
    for (int idx : feat_ids_to_verify) {
      const int match_idx = feature_source_map[idx];
      verify_features.push_back(features2[idx]);
      verify_features_tracked.push_back(features1[match_idx]);
    }

    tracking_flags |= cv::OPTFLOW_USE_INITIAL_FLOW;

    // Unused track error.
    std::vector<float> verify_track_error(num_to_verify);
    feature_status_.resize(num_to_verify);

    if (use_cv_tracking_) {
#if CV_MAJOR_VERSION == 3
      cv::calcOpticalFlowPyrLK(input_frame2, input_frame1, verify_features,
                               verify_features_tracked, feature_status_,
                               verify_track_error, cv_window_size,
                               pyramid_levels_, cv_criteria, tracking_flags);
#endif
    } else {
      LOG(ERROR) << "only cv tracking is supported.";
      return;
    }

    // Check feature destinations, that when tracked back to from data1 to
    // data2, don't differ more than a threshold from their original location
    // in data1.
    std::vector<uchar> verify_result(num_valid_features, 1);  // 1 for accept.
    int num_accepted = 0;
    for (int k = 0; k < num_to_verify; ++k) {
      const int idx = feat_ids_to_verify[k];
      const int match_idx = feature_source_map[idx];
      const cv::Point2f diff =
          features1[match_idx] - verify_features_tracked[k];
      const float dist = std::sqrt(diff.dot(diff));
      verify_distance[idx] = dist;
      verify_result[idx] =
          dist < options_.verification_distance() && feature_status_[k] == 1;
      num_accepted += verify_result[idx];
    }
    VLOG(1) << "Accepted number of verified features " << num_accepted;

    num_valid_features = InplaceFeatureSelection(
        to_data_ptr, {&feature_source_map},
        {&feature_track_error_, &verify_distance},
        [this, verify_result](int i) -> bool { return verify_result[i]; });
  }

  if (frac_long_features_rejected != nullptr) {
    *frac_long_features_rejected = 0;
  }

  // Verify long features if requested.
  if (IsVerifyLongFeatures() && num_valid_features > 0) {
    // track_win_size > 0, checked above.
    const float denom = 1.0f / (track_win_size * track_win_size * 255.0);
    int new_tracks = 0;
    for (int track_id : data2.track_ids) {
      if (track_id > prev_id_threshold) {
        ++new_tracks;
      }
    }

    // Select features that don't differ from the original extracted path by
    // more than a threshold. If that is the case, propagate original patch from
    // data1 to data2. Used for subsequent verifications.
    int num_selected_features = InplaceFeatureSelection(
        to_data_ptr, {&feature_source_map},
        {&feature_track_error_, &verify_distance},
        [this, denom, &data1, &data2, &feature_source_map](int i) -> bool {
          cv::Mat& mat1 = (*data1.neighborhoods)[feature_source_map[i]];
          cv::Mat& mat2 = (*data2.neighborhoods)[i];
          const float norm = cv::norm(mat1, mat2, cv::NORM_L1) * denom;
          if (norm < options_.long_feature_verification_threshold()) {
            // Store original patch.
            mat2 = mat1;
            return true;
          } else {
            return false;
          }
        });

    if (frac_long_features_rejected) {
      // There needs to be a reasonable number of features to test this
      // criteria.
      constexpr int kMinPrevValidFeatures = 10;
      if (num_valid_features > kMinPrevValidFeatures) {
        *frac_long_features_rejected =
            (1.0f - num_selected_features * (1.0f / num_valid_features));
        // Note: One could add number of new tracks here, i.e.
        // += new_tracks / num_valid_features.
        // Concern is that is very shaky video where a lot of tracks get created
        // this criteria overpowers the current one.
      }
    }
    num_valid_features = num_selected_features;
  }

  // How many tracking steps were used to generate these features.
  data2.last_feature_extraction_time = 1 + data1.last_feature_extraction_time;
  // Features are resized, so they cannot be considered initialized.
  data2.features_initialized = false;

  // Copy verified features to results.
  results.reserve(num_valid_features);
  for (int i = 0; i < num_valid_features; ++i) {
    const int match_idx = feature_source_map[i];
    const Vector2_f point1 =
        Vector2_f(features1[match_idx].x, features1[match_idx].y) *
        downsample_scale_;
    const Vector2_f point2 =
        Vector2_f(features2[i].x, features2[i].y) * downsample_scale_;

    if (PointOutOfBound(point1, original_width_, original_height_) ||
        PointOutOfBound(point2, original_width_, original_height_)) {
      continue;
    }

    const Vector2_f flow = point2 - point1;
    results.emplace_back(point1, flow, feature_track_error_[i],
                         corner_responses2[i], octaves2[i], track_ids2[i],
                         verify_distance[i]);

    if (long_track_data_ != nullptr && track_ids1[match_idx] != track_ids2[i]) {
      results.back().flags |= RegionFlowFeature::FLAG_BROKEN_TRACK;
    }

    if (IsVerifyLongFeatures()) {
      const cv::Mat& orig_patch = (*data1.neighborhoods)[match_idx];
      results.back().orig_neighborhood.reset(new cv::Mat(orig_patch));
    }

    if (data1.orb.computed) {
      results.back().descriptors = data1.orb.descriptors.row(match_idx);
    }
  }
}

void RegionFlowComputation::AppendUniqueFeaturesSorted(
    const TrackedFeatureView& to_be_added, TrackedFeatureView* features) const {
  for (auto feature_ptr : to_be_added) {
    auto insert_pos =
        std::lower_bound(features->begin(), features->end(), feature_ptr);
    if (insert_pos == features->end() || *insert_pos != feature_ptr) {
      // Not present yet.
      features->insert(insert_pos, feature_ptr);
      feature_ptr->irls_weight = 1;
    }

    // Present, increase score.
    ++feature_ptr->irls_weight;
  }
}

void RegionFlowComputation::InitializeFeatureLocationsFromTransform(
    int from, int to, const Homography& transform) {
  const int index1 = data_queue_.size() + from - 1;
  const int index2 = data_queue_.size() + to - 1;
  FrameTrackingData& data1 = *data_queue_[index1];
  FrameTrackingData* data2 = data_queue_[index2].get();

  data2->features = data1.features;

  for (cv::Point2f& feature : data2->features) {
    const Vector2_f trans_pt =
        TransformPoint(transform, Vector2_f(feature.x, feature.y));
    feature = cv::Point2f(trans_pt.x(), trans_pt.y());
  }

  data2->source = &data1;
  data2->features_initialized = true;
}

void RegionFlowComputation::InitializeFeatureLocationsFromPreviousResult(
    int from, int to) {
  CHECK_NE(from, to) << "Cannot initialize FrameTrackingData from itself.";

  const int index1 = data_queue_.size() + from - 1;
  const int index2 = data_queue_.size() + to - 1;
  CHECK_GE(index1, 0);
  CHECK_LT(index1, data_queue_.size());
  CHECK_GE(index2, 0);
  CHECK_LT(index2, data_queue_.size());
  const FrameTrackingData& data1 = *data_queue_[index1];
  FrameTrackingData* data2 = data_queue_[index2].get();
  CHECK(data1.source != nullptr);

  if (!data1.features_initialized) {
    data2->features = data1.source->features;
    for (int k = 0; k < data1.feature_source_map.size(); ++k) {
      data2->features[data1.feature_source_map[k]] = data1.features[k];
    }
  } else {
    data2->features = data1.features;
    CHECK_EQ(data1.features.size(), data1.source->features.size());
  }
  data2->source = data1.source;
  data2->features_initialized = true;
}

namespace {

void ComputeMeanForRegionFlow(RegionFlowFrame::RegionFlow* region_flow) {
  // Compute mean for this region.
  Vector2_f centroid(0, 0);
  Vector2_f mean_flow(0, 0);

  for (const auto& feature : region_flow->feature()) {
    centroid += Vector2_f(feature.x(), feature.y());
    mean_flow += Vector2_f(feature.dx(), feature.dy());
  }

  const float denom = 1.0f / region_flow->feature_size();
  centroid *= denom;
  mean_flow *= denom;

  region_flow->set_centroid_x(centroid.x());
  region_flow->set_centroid_y(centroid.y());
  region_flow->set_flow_x(mean_flow.x());
  region_flow->set_flow_y(mean_flow.y());
}

}  // namespace.

void RegionFlowComputation::ComputeBlockBasedFlow(
    TrackedFeatureList* feature_list,
    TrackedFeatureView* inlier_features) const {
  MEASURE_TIME << "Block based flow";
  TrackedFeatureView inlier_view;
  inlier_view.reserve(feature_list->size());

  const float frame_diam = hypot(original_width_, original_height_);
  const float max_magnitude_threshold =
      frame_diam * options_.max_magnitude_threshold_ratio();
  float sq_max_magnitude_threshold =
      max_magnitude_threshold * max_magnitude_threshold;

  // Refine threshold, for magnitudes over N times the mean motion, if
  // requested.
  if (!feature_list->empty() && options_.median_magnitude_bounds() > 0) {
    std::vector<float> motion_magnitudes;
    motion_magnitudes.reserve(feature_list->size());
    for (const auto& feature : *feature_list) {
      motion_magnitudes.push_back(feature.flow.Norm2());
    }
    auto median_iter = motion_magnitudes.begin() + motion_magnitudes.size() / 2;
    std::nth_element(motion_magnitudes.begin(), median_iter,
                     motion_magnitudes.end());
    const float median = *median_iter;
    // Only apply if non-zero motion is present (at least one pixel).
    if (median > 1.0) {
      const float outlier_threshold = median *
                                      options_.median_magnitude_bounds() *
                                      options_.median_magnitude_bounds();
      // Refine threshold.
      sq_max_magnitude_threshold =
          min(sq_max_magnitude_threshold, outlier_threshold);
    }
  }

  for (auto& feature : *feature_list) {
    // Skip features with exceeding motions.
    if (feature.flow.Norm2() < sq_max_magnitude_threshold) {
      inlier_view.push_back(&feature);
      inlier_view.back()->num_bins = 0;
    }
  }

  const int num_overlaps = options_.fast_estimation_overlap_grids();
  const int num_grids = block_levels_ * num_overlaps * num_overlaps;

  // Put all features into region bins.
  std::vector<TrackedFeatureMap> grid_feature_views(num_grids);

  int grid_idx = 0;
  int block_width = block_width_;
  int block_height = block_height_;

  for (int level = 0; level < block_levels_; ++level) {
    const float inv_block_width = 1.0f / block_width;
    const float inv_block_height = 1.0f / block_height;

    for (int overlap_y = 0; overlap_y < num_overlaps; ++overlap_y) {
      // |    |    |    |  <- unshifted
      // | |    |    |  |  <- shifted
      //   ^
      //   block_height * overlap_y / num_overlaps  := t
      // We need to add block_height - t to each value, so that at t
      // we get bin 1.
      // Special case is overlap_y == 0 -> shift = 0.
      const int grid_shift_y =
          overlap_y == 0
              ? 0
              : (block_height - block_height * overlap_y / num_overlaps);

      for (int overlap_x = 0; overlap_x < num_overlaps;
           ++overlap_x, ++grid_idx) {
        const int grid_shift_x =
            overlap_x == 0
                ? 0
                : (block_width - block_width * overlap_x / num_overlaps);

        const int bins_per_row =
            std::ceil((original_width_ + grid_shift_x) * inv_block_width);
        const int bins_per_column =
            std::ceil((original_height_ + grid_shift_y) * inv_block_height);
        TrackedFeatureMap& feature_view = grid_feature_views[grid_idx];
        feature_view.resize(bins_per_row * bins_per_column);

        for (auto feature_ptr : inlier_view) {
          const int x = feature_ptr->point.x() + 0.5f + grid_shift_x;
          const int y = feature_ptr->point.y() + 0.5f + grid_shift_y;
          const int block_x = x * inv_block_width;
          const int block_y = y * inv_block_height;

          int block_id = block_y * bins_per_row + block_x;
          feature_view[block_id].push_back(feature_ptr);
        }
      }
    }

    // We use smallest block width and height later on.
    if (level + 1 < block_levels_) {
      block_width = (block_width + 1) / 2;
      block_height = (block_height + 1) / 2;
    }
  }

  for (int k = 0; k < num_grids; ++k) {
    TrackedFeatureMap& region_features = grid_feature_views[k];
    const int min_inliers = GetMinNumFeatureInliers(region_features);
    for (TrackedFeatureView& feature_view : region_features) {
      if (feature_view.size() >= min_inliers) {
        for (auto feature_ptr : feature_view) {
          ++feature_ptr->num_bins;
        }
      } else {
        feature_view.clear();
      }
    }
  }

  if (num_grids == 1) {
    TrackedFeatureView all_inliers;
    DetermineRegionFlowInliers(grid_feature_views[0], &all_inliers);
    AppendUniqueFeaturesSorted(all_inliers, inlier_features);
  } else {
    // Get all features across all grids without duplicates.
    std::vector<TrackedFeatureView> grid_inliers(num_grids);
    ParallelFor(
        0, num_grids, 1,
        [&grid_feature_views, &grid_inliers, this](const BlockedRange& range) {
          for (int k = range.begin(); k < range.end(); ++k) {
            grid_inliers[k].reserve(grid_feature_views[k].size());
            DetermineRegionFlowInliers(grid_feature_views[k], &grid_inliers[k]);
          }
        });

    for (int grid = 0; grid < num_grids; ++grid) {
      AppendUniqueFeaturesSorted(grid_inliers[grid], inlier_features);
    }
  }
}

void RegionFlowComputation::DetermineRegionFlowInliers(
    const TrackedFeatureMap& region_feature_map,
    TrackedFeatureView* inliers) const {
  CHECK(inliers);
  inliers->clear();

  // Run RANSAC on each region.
  const int max_iterations = options_.ransac_rounds_per_region();
  float absolute_err_threshold =
      max<float>(options_.absolute_inlier_error_threshold(),
                 options_.frac_inlier_error_threshold() *
                     hypot(original_width_, original_height_));
  absolute_err_threshold *= absolute_err_threshold;

  // Save ptr to each inlier feature.
  TrackedFeatureView inlier_set;
  TrackedFeatureView best_inlier_set;
  unsigned int seed = 900913;  // = Google in leet :)
  std::default_random_engine rand_gen(seed);

  const int min_features = GetMinNumFeatureInliers(region_feature_map);

  for (const TrackedFeatureView& region_features : region_feature_map) {
    if (region_features.empty()) {
      continue;
    }
    // Select top N inlier sets.
    int loop_count = options_.top_inlier_sets();

    // Get local copy if necessary and sort.
    const TrackedFeatureView* all_features = nullptr;
    TrackedFeatureView all_features_storage;

    if (loop_count > 1) {
      all_features_storage = region_features;
      all_features = &all_features_storage;
      std::sort(all_features_storage.begin(), all_features_storage.end());
    } else {
      all_features = &region_features;
    }

    const int num_features = all_features->size();

    // Extract inlier sets as long as more than 20% of original features remain.
    int last_inlier_set_size = 0;

    while (all_features->size() >= max(min_features, num_features / 5) &&
           loop_count-- > 0) {
      best_inlier_set.clear();
      std::uniform_int_distribution<> distribution(0, all_features->size() - 1);

      for (int i = 0; i < max_iterations; ++i) {
        // Pick a random vector.
        const int rand_idx = distribution(rand_gen);
        Vector2_f vec = (*all_features)[rand_idx]->flow;

        float relative_err_threshold =
            options_.relative_inlier_error_threshold() * vec.Norm();
        relative_err_threshold *= relative_err_threshold;
        const float err_threshold =
            std::max(relative_err_threshold, absolute_err_threshold);

        // Determine inlier vectors.
        inlier_set.clear();

        for (auto feature_ptr : *all_features) {
          if ((feature_ptr->flow - vec).Norm2() < err_threshold) {
            inlier_set.push_back(feature_ptr);
          }
        }

        if (inlier_set.size() >= best_inlier_set.size()) {
          best_inlier_set.swap(inlier_set);
        }
      }

      if (best_inlier_set.size() >=
          max(options_.min_feature_inliers(), last_inlier_set_size / 2)) {
        last_inlier_set_size = best_inlier_set.size();
        inliers->insert(inliers->end(), best_inlier_set.begin(),
                        best_inlier_set.end());

        if (loop_count > 0) {
          // Remove inliers from all feature set.
          TrackedFeatureView remaining_features;
          std::set_difference(all_features_storage.begin(),
                              all_features_storage.end(),
                              best_inlier_set.begin(), best_inlier_set.end(),
                              back_inserter(remaining_features));
          all_features_storage.swap(remaining_features);
        }
      } else {
        break;
      }
    }
  }  // end traverse region bins.
}

int RegionFlowComputation::GetMinNumFeatureInliers(
    const TrackedFeatureMap& region_feature_map) const {
  // Determine number of features.
  int total_features = 0;
  for (const TrackedFeatureView& region_features : region_feature_map) {
    total_features += region_features.size();
  }

  CHECK(!region_feature_map.empty())
      << "Empty grid passed. Check input dimensions";

  const float threshold =
      std::max<int>(options_.min_feature_inliers(),
                    options_.relative_min_feature_inliers() * total_features /
                        region_feature_map.size());

  return threshold;
}

void RegionFlowComputation::RegionFlowFeatureListToRegionFlow(
    const RegionFlowFeatureList& feature_list, RegionFlowFrame* frame) const {
  CHECK(frame != nullptr);

  frame->set_num_total_features(feature_list.feature_size());
  frame->set_unstable_frame(feature_list.unstable());
  if (feature_list.has_blur_score()) {
    frame->set_blur_score(feature_list.blur_score());
  }
  frame->set_frame_width(feature_list.frame_width());
  frame->set_frame_height(feature_list.frame_height());

  RegionFlowFrame::BlockDescriptor* block_descriptor =
      frame->mutable_block_descriptor();

  // Compute minimum block size.
  int min_block_width = block_width_;
  int min_block_height = block_height_;

  for (int level = 0; level < block_levels_; ++level) {
    if (level + 1 < block_levels_) {
      min_block_width = (min_block_width + 1) / 2;
      min_block_height = (min_block_height + 1) / 2;
    }
  }
  block_descriptor->set_block_width(min_block_width);
  block_descriptor->set_block_height(min_block_height);
  const int bins_per_row =
      std::ceil(original_width_ * (1.0f / min_block_width));
  const int bins_per_col =
      std::ceil(original_height_ * (1.0f / min_block_height));
  block_descriptor->set_num_blocks_x(bins_per_row);
  block_descriptor->set_num_blocks_y(bins_per_col);

  const int num_regions = bins_per_row * bins_per_col;
  frame->mutable_region_flow()->Reserve(num_regions);
  for (int k = 0; k < num_regions; ++k) {
    frame->add_region_flow()->set_region_id(k);
  }

  // Add feature according smallest block width and height to regions.
  for (const auto& feature : feature_list.feature()) {
    const int x = static_cast<int>(feature.x());
    const int y = static_cast<int>(feature.y());
    // Guard, in case equation is wrong.
    int region_id = min(
        num_regions, y / min_block_height * bins_per_row + x / min_block_width);
    *frame->mutable_region_flow(region_id)->add_feature() = feature;
  }

  for (auto& region_flow : *frame->mutable_region_flow()) {
    ComputeMeanForRegionFlow(&region_flow);
  }
}

void RegionFlowComputation::InitializeRegionFlowFeatureList(
    RegionFlowFeatureList* region_flow_feature_list) const {
  region_flow_feature_list->set_frame_width(original_width_);
  region_flow_feature_list->set_frame_height(original_height_);
  if (curr_blur_score_ >= 0.0f) {
    region_flow_feature_list->set_blur_score(curr_blur_score_);
  }

  region_flow_feature_list->set_distance_from_border(std::max(
      options_.patch_descriptor_radius(), options_.distance_from_border()));
  region_flow_feature_list->set_long_tracks(long_track_data_ != nullptr);
}

namespace {

bool IsPointWithinBounds(const Vector2_f& pt, int bounds, int frame_width,
                         int frame_height) {
  // Ensure stability under float -> rounding operations.
  if (pt.x() - 0.5f >= bounds && pt.x() + 0.5f <= frame_width - 1 - bounds &&
      pt.y() - 0.5f >= bounds && pt.y() + 0.5f <= frame_height - 1 - bounds) {
    return true;
  } else {
    return false;
  }
}

}  // namespace.

float RegionFlowComputation::TrackedFeatureViewToRegionFlowFeatureList(
    const TrackedFeatureView& region_feature_view,
    TrackedFeatureList* flattened_feature_list,
    RegionFlowFeatureList* region_flow_feature_list) const {
  const int border = region_flow_feature_list->distance_from_border();

  region_flow_feature_list->mutable_feature()->Reserve(
      region_feature_view.size());

  float sq_flow_sum = 0;

  for (auto feature_ptr : region_feature_view) {
    const Vector2_f& location = feature_ptr->point;
    const Vector2_f& match_location = feature_ptr->point + feature_ptr->flow;

    if (border > 0) {
      if (!IsPointWithinBounds(location, border, original_width_,
                               original_height_) ||
          !IsPointWithinBounds(match_location, border, original_width_,
                               original_height_)) {
        continue;
      }
    }

    const Vector2_f& flow = feature_ptr->flow;
    sq_flow_sum += flow.Norm2();

    RegionFlowFeature* feature = region_flow_feature_list->add_feature();
    feature->set_x(location.x());
    feature->set_y(location.y());
    feature->set_dx(flow.x());
    feature->set_dy(flow.y());
    feature->set_tracking_error(feature_ptr->tracking_error);
    feature->set_corner_response(feature_ptr->corner_response);

    if (long_track_data_ != nullptr) {
      feature->set_track_id(feature_ptr->track_id);
    }
    feature->set_flags(feature_ptr->flags);

    switch (options_.irls_initialization()) {
      case RegionFlowComputationOptions::INIT_UNIFORM:
        feature->set_irls_weight(1.0f);
        break;

      case RegionFlowComputationOptions::INIT_CONSISTENCY:
        feature->set_irls_weight(2.0f * feature_ptr->irls_weight /
                                 feature_ptr->num_bins);
        break;
    }

    // Remember original TrackedFeature if requested.
    if (flattened_feature_list) {
      flattened_feature_list->push_back(*feature_ptr);
    }

    if (feature_ptr->descriptors.cols != 0) {
      feature->mutable_binary_feature_descriptor()->set_data(
          static_cast<void*>(feature_ptr->descriptors.data),
          feature_ptr->descriptors.cols);
    }
  }

  const int num_features = region_flow_feature_list->feature_size();
  float avg_motion = 0;
  if (num_features > 0) {
    avg_motion = std::sqrt(sq_flow_sum / num_features);
    if (avg_motion < kZeroMotion) {
      region_flow_feature_list->set_is_duplicated(true);
    }
  }

  return avg_motion;
}

// Check if enough features as specified by
// MotionStabilizationOptions::min_feature_requirement are present, and if
// features cover a sufficient area.
bool RegionFlowComputation::HasSufficientFeatures(
    const RegionFlowFeatureList& feature_list) {
  const int area_size = options_.min_feature_cover_grid();
  const float scaled_width = static_cast<float>(area_size) / original_width_;
  const float scaled_height = static_cast<float>(area_size) / original_height_;
  std::vector<int> area_mask(area_size * area_size);

  for (const auto& feature : feature_list.feature()) {
    int x = feature.x() * scaled_width;
    int y = feature.y() * scaled_height;
    area_mask[y * area_size + x] = 1;
  }

  int covered_bins = std::accumulate(area_mask.begin(), area_mask.end(), 0);
  float area_covered =
      static_cast<float>(covered_bins) / (area_size * area_size);

  const int num_features = feature_list.feature_size();
  bool has_sufficient_features =
      num_features >= options_.min_feature_requirement() &&
      area_covered > options_.min_feature_cover();

  if (has_sufficient_features) {
    VLOG(1) << "Sufficient features: " << num_features;
  } else {
    VLOG(1) << "!! Insufficient features: " << num_features
            << " required: " << options_.min_feature_requirement()
            << " cover: " << area_covered
            << " required: " << options_.min_feature_cover();
  }

  VLOG(1) << (has_sufficient_features ? "Has sufficient " : "Insufficient ")
          << " features: " << num_features;

  return has_sufficient_features;
}

int RegionFlowComputation::PyramidLevelsFromTrackDistance(
    float track_distance) {
  // The number of pyramid levels l is chosen such that
  // 2^l * tracking_window_size / 2 >= track_distance.
  int pyramid_levels =
      std::ceil(std::log2(std::max(track_distance, 1.0f) * 2.f /
                          options_.tracking_options().tracking_window_size()));
  // The maximum pyramid level that we are targeting. The minimum size in the
  // pyramid is intended to be 2x2.
  const int max_pyramid_levels =
      max<int>(1, std::log2(min<float>(frame_height_, frame_width_)) - 1);
  pyramid_levels = min(max_pyramid_levels, max(pyramid_levels, 2));
  return pyramid_levels;
}

void RegionFlowComputation::ComputeBlurMask(const cv::Mat& input,
                                            cv::Mat* min_eig_vals,
                                            cv::Mat* mask) {
  MEASURE_TIME << "Computing blur score";
  const auto& blur_options = options_.blur_score_options();
  cv::cornerMinEigenVal(input, *corner_values_, 3);

  // Create over-exposure mask to mask out corners in high exposed areas.
  // Reason is, that motion blur does not affect lights in the same manner as
  // diffuse surfaces due to the limited dynamic range of the camera.
  // Blurring a light tends not to lower the corner score but simply changes the
  // shape of the light with the corner score being unaffected.
  cv::compare(input, 245, *corner_mask_, cv::CMP_GE);

  // Dilate reads out of bound values.
  if (corner_mask_->rows > 5 && corner_mask_->cols > 5) {
    cv::Mat dilate_domain(*corner_mask_, cv::Range(2, corner_mask_->rows - 2),
                          cv::Range(2, corner_mask_->cols - 2));
    cv::Mat kernel(5, 5, CV_8U);
    kernel.setTo(1.0);
    cv::dilate(dilate_domain, dilate_domain, kernel);
  }
  corner_values_->setTo(cv::Scalar(0), *corner_mask_);

  // Box filter corner score to diffuse and suppress noise.
  cv::boxFilter(
      *corner_values_, *corner_filtered_, CV_32F,
      cv::Size(blur_options.box_filter_diam(), blur_options.box_filter_diam()));

  // Determine maximum cornerness in robust manner over bins.
  // Specular reflections or colored lights might not be filtered out by
  // the above operation.
  const int max_blocks = 8;
  const int block_width =
      std::ceil(static_cast<float>(corner_filtered_->cols) / max_blocks);
  const int block_height =
      std::ceil(static_cast<float>(corner_filtered_->rows) / max_blocks);
  std::vector<float> block_maximums;
  for (int block_y = 0; block_y < max_blocks; ++block_y) {
    if (block_y * block_height >= corner_filtered_->rows) {
      continue;
    }

    cv::Range y_range(block_y * block_height, min((block_y + 1) * block_height,
                                                  corner_filtered_->rows));

    for (int block_x = 0; block_x < max_blocks; ++block_x) {
      if (block_x * block_width >= corner_filtered_->cols) {
        continue;
      }

      cv::Range x_range(block_x * block_width, min((block_x + 1) * block_width,
                                                   corner_filtered_->cols));
      cv::Mat block(*corner_filtered_, y_range, x_range);
      double min_val, max_val;
      cv::minMaxLoc(block, &min_val, &max_val);
      block_maximums.push_back(max_val);
    }
  }

  auto block_median =
      block_maximums.begin() + static_cast<int>(block_maximums.size() * 0.75);
  std::nth_element(block_maximums.begin(), block_median, block_maximums.end());
  const float max_val = *block_median;

  const float thresh =
      max<float>(blur_options.absolute_cornerness_threshold(),
                 blur_options.relative_cornerness_threshold() * max_val);

  cv::compare(*corner_filtered_, thresh, *corner_mask_, cv::CMP_GE);
}

float RegionFlowComputation::ComputeBlurScore(const cv::Mat& input) {
  // TODO: Already computed during feature extraction, re-use!
  cv::cornerMinEigenVal(input, *corner_values_, 3);
  ComputeBlurMask(input, corner_values_.get(), corner_mask_.get());

  // Compute median corner score over masked area.
  std::vector<float> corner_score;
  corner_score.reserve(frame_width_ * frame_height_);
  for (int i = 0; i < corner_mask_->rows; ++i) {
    const uint8_t* mask_ptr = corner_mask_->ptr<uint8_t>(i);
    const float* corner_ptr = corner_values_->ptr<float>(i);
    for (int j = 0; j < corner_mask_->cols; ++j) {
      if (mask_ptr[j]) {
        corner_score.push_back(corner_ptr[j]);
      }
    }
  }

  const auto& blur_options = options_.blur_score_options();
  std::vector<float>::iterator median_iter =
      corner_score.begin() +
      static_cast<int>(corner_score.size() * blur_options.median_percentile());

  float blur_score = 1e10f;
  if (median_iter != corner_score.end()) {
    std::nth_element(corner_score.begin(), median_iter, corner_score.end());
    if (*median_iter > 1e-10f) {
      blur_score = 1.0f / *median_iter;
    }
  }

  return blur_score;
}

}  // namespace mediapipe
