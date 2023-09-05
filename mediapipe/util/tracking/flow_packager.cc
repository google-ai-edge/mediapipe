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

#include "mediapipe/util/tracking/flow_packager.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/tracking/camera_motion.h"
#include "mediapipe/util/tracking/camera_motion.pb.h"
#include "mediapipe/util/tracking/motion_estimation.h"
#include "mediapipe/util/tracking/motion_models.h"
#include "mediapipe/util/tracking/motion_models.pb.h"
#include "mediapipe/util/tracking/region_flow.pb.h"

namespace mediapipe {

FlowPackager::FlowPackager(const FlowPackagerOptions& options)
    : options_(options) {
  if (options_.binary_tracking_data_support()) {
    ABSL_CHECK_LE(options.domain_width(), 256);
    ABSL_CHECK_LE(options.domain_height(), 256);
  }
}

namespace {

// Performs rounding of float vector position to int.
class FeatureIntegerPosition {
 public:
  // Scales a feature's location in x and y by scale_x and scale_y respectively.
  // Limits feature position to the integer domain
  // [0, width - 1] x [0, height - 1]
  FeatureIntegerPosition(float scale_x, float scale_y, int width, int height)
      : scale_x_(scale_x), scale_y_(scale_y), width_(width), height_(height) {}

  Vector2_i ToIntPosition(const RegionFlowFeature& feature) const {
    return Vector2_i(
        std::max(0, std::min<int>(width_ - 1, feature.x() * scale_x_ + 0.5f)),
        std::max(0, std::min<int>(height_ - 1, feature.y() * scale_y_ + 0.5f)));
  }

 private:
  float scale_x_;
  float scale_y_;
  int width_;
  int height_;
};

// Lexicographic compare (first in x, then in y) under scaled integer rounding
// as specified by FeatureIntegerPosition.
class IntegerColumnComparator {
 public:
  IntegerColumnComparator(float scale_x, float scale_y, int width, int height)
      : integer_pos_(scale_x, scale_y, width, height) {}

  bool operator()(const RegionFlowFeature& lhs,
                  const RegionFlowFeature& rhs) const {
    const Vector2_i vec_lhs = integer_pos_.ToIntPosition(lhs);
    const Vector2_i vec_rhs = integer_pos_.ToIntPosition(rhs);
    return (vec_lhs.x() < vec_rhs.x()) ||
           (vec_lhs.x() == vec_rhs.x() && vec_lhs.y() < vec_rhs.y());
  }

 private:
  const FeatureIntegerPosition integer_pos_;
};

template <typename T>
inline std::string EncodeToString(const T& value) {
  std::string s(sizeof(T), 0);
  memcpy(&s[0], &value, sizeof(T));
  return s;
}

template <typename T>
inline std::string EncodeVectorToString(const std::vector<T>& vec) {
  std::string s(vec.size() * sizeof(T), 0);
  typename std::vector<T>::const_iterator iter;
  char* ptr;
  for (iter = vec.begin(), ptr = &s[0]; iter != vec.end();
       ++iter, ptr += sizeof(T)) {
    memcpy(ptr, &(*iter), sizeof(T));
  }
  return s;
}

template <typename T>
inline bool DecodeFromStringView(absl::string_view str, T* result) {
  ABSL_CHECK(result != nullptr);
  if (sizeof(*result) != str.size()) {
    return false;
  }
  memcpy(result, str.data(), sizeof(T));
  return true;
}

template <typename T>
inline bool DecodeVectorFromStringView(absl::string_view str,
                                       std::vector<T>* result) {
  ABSL_CHECK(result != nullptr);
  if (str.size() % sizeof(T) != 0) return false;
  result->clear();
  result->reserve(str.size() / sizeof(T));
  T value;
  const char* begin = str.data();
  const char* end = str.data() + str.size();
  for (const char* ptr = begin; ptr != end; ptr += sizeof(T)) {
    memcpy(&value, ptr, sizeof(T));
    result->push_back(value);
  }
  return true;
}
}  // namespace.

void FlowPackager::PackFlow(const RegionFlowFeatureList& feature_list,
                            const CameraMotion* camera_motion,
                            TrackingData* tracking_data) const {
  ABSL_CHECK(tracking_data);
  ABSL_CHECK_GT(feature_list.frame_width(), 0);
  ABSL_CHECK_GT(feature_list.frame_height(), 0);

  // Scale flow to output domain.
  const float dim_x_scale =
      options_.domain_width() * (1.0f / feature_list.frame_width());
  const float dim_y_scale =
      options_.domain_height() * (1.0f / feature_list.frame_height());

  const bool long_tracks = feature_list.long_tracks();

  // Sort features lexicographically.
  RegionFlowFeatureList sorted_feature_list(feature_list);

  SortRegionFlowFeatureList(dim_x_scale, dim_y_scale, &sorted_feature_list);

  tracking_data->set_domain_width(options_.domain_width());
  tracking_data->set_domain_height(options_.domain_height());
  tracking_data->set_frame_aspect(feature_list.frame_width() * 1.0f /
                                  feature_list.frame_height());
  tracking_data->set_global_feature_count(feature_list.feature_size());
  int flags = 0;

  if (camera_motion == nullptr ||
      camera_motion->type() > CameraMotion::UNSTABLE_SIM) {
    flags |= TrackingData::FLAG_BACKGROUND_UNSTABLE;
  } else {
    Homography transform;
    CameraMotionToHomography(*camera_motion, &transform);
    Homography normalization = HomographyAdapter::Embed(
        AffineAdapter::FromArgs(0, 0, dim_x_scale, 0, 0, dim_y_scale));
    Homography inv_normalization =
        HomographyAdapter::Embed(AffineAdapter::FromArgs(
            0, 0, 1.0f / dim_x_scale, 0, 0, 1.0f / dim_y_scale));
    *tracking_data->mutable_background_model() =
        ModelCompose3(normalization, transform, inv_normalization);
  }

  if (camera_motion != nullptr) {
    tracking_data->set_average_motion_magnitude(
        camera_motion->average_magnitude());
  }

  if (feature_list.is_duplicated()) {
    flags |= TrackingData::FLAG_DUPLICATED;
  }
  tracking_data->set_frame_flags(flags);

  const int num_vectors = sorted_feature_list.feature_size();
  TrackingData::MotionData* data = tracking_data->mutable_motion_data();
  data->set_num_elements(num_vectors);

  // Initialize col starts with "unseen" marker.
  std::vector<float> col_start(options_.domain_width() + 1, -1);

  int last_col = -1;
  int last_row = -1;
  FeatureIntegerPosition integer_pos(dim_x_scale, dim_y_scale,
                                     options_.domain_width(),
                                     options_.domain_height());

  // Store feature and corresponding motion (minus camera motion) in
  // compressed sparse column format:
  // https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_.28CSC_or_CCS.29
  for (const auto& feature : sorted_feature_list.feature()) {
    float flow_x = feature.dx() * dim_x_scale;
    float flow_y = feature.dy() * dim_y_scale;
    Vector2_i loc = integer_pos.ToIntPosition(feature);

    // Convert back to float for accurate background model computation.
    Vector2_f loc_f = Vector2_f::Cast(loc);

    if (camera_motion) {
      Vector2_f residual = HomographyAdapter::TransformPoint(
                               tracking_data->background_model(), loc_f) -
                           loc_f;
      flow_x -= residual.x();
      flow_y -= residual.y();
    }

    data->add_vector_data(flow_x);
    data->add_vector_data(flow_y);
    data->add_row_indices(loc.y());

    if (feature.has_binary_feature_descriptor()) {
      data->add_feature_descriptors()->set_data(
          feature.binary_feature_descriptor().data());
    }

    if (long_tracks) {
      data->add_track_id(feature.track_id());
    }

    const int curr_col = loc.x();

    if (curr_col != last_col) {
      ABSL_CHECK_LT(last_col, curr_col);
      ABSL_CHECK_EQ(-1, col_start[curr_col]);
      col_start[curr_col] = data->row_indices_size() - 1;
      last_col = curr_col;
    } else {
      ABSL_CHECK_LE(last_row, loc.y());
    }
    last_row = loc.y();
  }

  col_start[0] = 0;
  col_start[options_.domain_width()] = num_vectors;

  // Fill unset values with previously set value. Propagate end value.
  for (int i = options_.domain_width() - 1; i > 0; --i) {
    if (col_start[i] < 0) {
      ABSL_DCHECK_GE(col_start[i + 1], 0);
      col_start[i] = col_start[i + 1];
    }
  }

  for (const auto& col_idx : col_start) {
    data->add_col_starts(col_idx);
  }

  // Check monotonicity of the row indices.
  for (int c = 0; c < options_.domain_width(); ++c) {
    const int r_start = data->col_starts(c);
    const int r_end = data->col_starts(c + 1);
    for (int r = r_start; r < r_end - 1; ++r) {
      ABSL_CHECK_LE(data->row_indices(r), data->row_indices(r + 1));
    }
  }

  ABSL_CHECK_EQ(data->vector_data_size(), 2 * data->row_indices_size());

  *data->mutable_actively_discarded_tracked_ids() =
      feature_list.actively_discarded_tracked_ids();
}

void FlowPackager::EncodeTrackingData(const TrackingData& tracking_data,
                                      BinaryTrackingData* binary_data) const {
  ABSL_CHECK(options_.binary_tracking_data_support());
  ABSL_CHECK(binary_data != nullptr);

  int32_t frame_flags = 0;
  const bool high_profile = options_.use_high_profile();
  if (high_profile) {
    frame_flags |= TrackingData::FLAG_PROFILE_HIGH;
  } else {
    frame_flags |= TrackingData::FLAG_PROFILE_BASELINE;  // No op.
  }

  if (options_.high_fidelity_16bit_encode()) {
    frame_flags |= TrackingData::FLAG_HIGH_FIDELITY_VECTORS;
  }

  // Copy background flag.
  frame_flags |=
      tracking_data.frame_flags() & TrackingData::FLAG_BACKGROUND_UNSTABLE;

  const TrackingData::MotionData& motion_data = tracking_data.motion_data();
  int32_t num_vectors = motion_data.num_elements();

  // Compute maximum vector or delta vector value.
  float max_vector_value = 0;
  if (high_profile) {
    for (int k = 2; k < 2 * num_vectors; ++k) {
      max_vector_value = std::max<float>(
          max_vector_value,
          fabs(motion_data.vector_data(k) - motion_data.vector_data(k - 2)) *
              1.02f);  // Expand by 2% to account for
                       // rounding issues.
    }
  } else {
    for (const float vector_value : motion_data.vector_data()) {
      max_vector_value = std::max<float>(max_vector_value, fabs(vector_value));
    }
  }

  const int32_t domain_width = tracking_data.domain_width();
  const int32_t domain_height = tracking_data.domain_height();
  ABSL_CHECK_LT(domain_height, 256) << "Only heights below 256 are supported.";
  const float frame_aspect = tracking_data.frame_aspect();

  // Limit vector value from above (to 20% frame diameter) and below (small
  // eps).
  const float max_vector_threshold = hypot(domain_width, domain_height) * 0.2f;
  // Warn if too much truncation.
  if (max_vector_value > max_vector_threshold * 1.5f) {
    ABSL_LOG(WARNING) << "A lot of truncation will occur during encoding. "
                      << "Vector magnitudes are larger than 20% of the "
                      << "frame diameter.";
  }

  max_vector_value =
      std::min<float>(max_vector_threshold, std::max(1e-4f, max_vector_value));

  // Compute scales for 16bit and 8bit float -> int conversion.
  // Use highest bit for sign.
  const int kByteMax16 = (1 << 15) - 1;
  const int kByteMax8 = (1 << 7) - 1;

  // Scale such that highest vector value is mapped to kByteMax
  int scale_16 = std::ceil(kByteMax16 / max_vector_value);
  int scale_8 = std::ceil(kByteMax8 / max_vector_value);

  const int32_t scale =
      options_.high_fidelity_16bit_encode() ? scale_16 : scale_8;
  const float inv_scale = 1.0f / scale;
  const int kByteMax =
      options_.high_fidelity_16bit_encode() ? kByteMax16 : kByteMax8;

  // Compressed flow to be encoded in binary format.
  std::vector<int16_t> flow_compressed_16;
  std::vector<int8_t> flow_compressed_8;

  flow_compressed_16.reserve(num_vectors);
  flow_compressed_8.reserve(num_vectors);

  std::vector<uint8_t> row_idx;
  row_idx.reserve(num_vectors);

  float average_error = 0;
  std::vector<int> col_starts(motion_data.col_starts().begin(),
                              motion_data.col_starts().end());

  // Separate both implementations for easier readability.
  // For details please refer to description in proto.
  // Low profile:
  //   * Encode vectors by scaling to integer format.
  //   * Keep sparse matrix format as is
  // High profile:
  //   * Encode deltas between vectors scaling them to integers
  //   * Re-use encoded vectors if delta is small, use ADVANCE flag in row
  //   index.
  //   * Delta encode row indices to reduce magnitude.
  //   * If two row deltas are small (< 8), encode in one byte
  if (!high_profile) {
    // Traverse columns.
    for (int c = 0; c < col_starts.size() - 1; ++c) {
      const int r_start = col_starts[c];
      const int r_end = col_starts[c + 1];
      for (int r = r_start; r < r_end; ++r) {
        const float flow_x_32f = motion_data.vector_data(2 * r);
        const float flow_y_32f = motion_data.vector_data(2 * r + 1);

        const int flow_x =
            std::max(-kByteMax, std::min<int>(kByteMax, flow_x_32f * scale));
        const int flow_y =
            std::max(-kByteMax, std::min<int>(kByteMax, flow_y_32f * scale));
        average_error += 0.5f * (fabs(flow_x * inv_scale - flow_x_32f) +
                                 fabs(flow_y * inv_scale - flow_y_32f));

        if (options_.high_fidelity_16bit_encode()) {
          flow_compressed_16.push_back(flow_x);
          flow_compressed_16.push_back(flow_y);
        } else {
          flow_compressed_8.push_back(flow_x);
          flow_compressed_8.push_back(flow_y);
        }

        ABSL_DCHECK_LT(motion_data.row_indices(r), 256);
        row_idx.push_back(motion_data.row_indices(r));
      }
    }
  } else {
    // Compress flow.
    int prev_flow_x = 0;
    int prev_flow_y = 0;
    const float reuse_threshold = options_.high_profile_reuse_threshold();
    int compressible = 0;

    std::vector<int> compressions_per_column(domain_width, 0);
    const int kAdvanceFlag = FlowPackagerOptions::ADVANCE_FLAG;
    const int kDoubleIndexEncode = FlowPackagerOptions::DOUBLE_INDEX_ENCODE;
    const int kIndexMask = FlowPackagerOptions::INDEX_MASK;

    // Traverse columns.
    for (int c = 0; c < motion_data.col_starts().size() - 1; ++c) {
      const int r_start = col_starts[c];
      const int r_end = col_starts[c + 1];
      for (int r = r_start; r < r_end; ++r) {
        int flow_x = 0;
        int flow_y = 0;
        bool advance = true;
        const float flow_x_32f = motion_data.vector_data(2 * r);
        const float flow_y_32f = motion_data.vector_data(2 * r + 1);

        // Delta coding of vectors.
        const float diff_x = flow_x_32f - prev_flow_x * inv_scale;
        const float diff_y = flow_y_32f - prev_flow_y * inv_scale;

        // Determine if previous flow can be re-used.
        if (fabs(diff_x) < reuse_threshold && fabs(diff_y) < reuse_threshold) {
          advance = false;
        } else {
          flow_x = std::max(-kByteMax, std::min<int>(kByteMax, diff_x * scale));

          flow_y = std::max(-kByteMax, std::min<int>(kByteMax, diff_y * scale));

          prev_flow_x += flow_x;
          prev_flow_y += flow_y;
        }

        average_error += 0.5f * (fabs(prev_flow_x * inv_scale - flow_x_32f) +
                                 fabs(prev_flow_y * inv_scale - flow_y_32f));

        // Combine into one 32 or 16 bit value (clear sign bits for the
        // right part before combining).
        if (advance) {
          if (options_.high_fidelity_16bit_encode()) {
            flow_compressed_16.push_back(flow_x);
            flow_compressed_16.push_back(flow_y);
          } else {
            flow_compressed_8.push_back(flow_x);
            flow_compressed_8.push_back(flow_y);
          }
        }

        // Delta code row indices in high profile mode and use two top bits
        // for status:
        // 10: single row encode, use next vector data.
        // (ADVANCE_FLAG)
        //
        // 11: double row encode: (3 bit + 3 bit = maximum of 7 + 7 row delta),
        // use next vector data for each.
        // (ADVANCE_FLAG | DOUBLE_INDEX_ENCODE)
        //
        // 00: single row encode + no advance (re-use previous vector data).
        // (no flags set)
        //
        // 01: double row encode + no advance (re-use previous vector data for
        // each).
        // (DOUBLE_INDEX_ENCODE)

        // Delta compress.
        int delta_row = motion_data.row_indices(r) -
                        (r == r_start ? 0 : motion_data.row_indices(r - 1));
        ABSL_CHECK_GE(delta_row, 0);

        bool combined = false;
        if (r > r_start) {
          int prev_row_idx = row_idx.back();
          if (!(prev_row_idx & kDoubleIndexEncode) &&      // Single encode.
              (prev_row_idx & kAdvanceFlag) == advance) {  // Same advance flag.
            // Both compressible (each index fits in 3 bit).
            if (delta_row < 8 && (prev_row_idx & kIndexMask) < 8) {
              // Encode two deltas into 6 bit.
              prev_row_idx = ((prev_row_idx & 0x07) << 3) | delta_row |
                             kDoubleIndexEncode | (advance ? kAdvanceFlag : 0);

              row_idx.back() = prev_row_idx;
              // Record as one compression for this column.
              ++compressions_per_column[c];
              ++compressible;
              combined = true;
            }
          }
        }

        if (!combined) {
          while (delta_row > kIndexMask) {
            // Special case of large displacement. Duplicate vector until sum of
            // deltas reaches target delta).
            row_idx.push_back(kIndexMask | (advance ? kAdvanceFlag : 0));
            delta_row -= kIndexMask;
            advance = false;  // Store same vector again, re-use previously
                              // encoded vector data.

            // Record as one addition for the column.
            --compressions_per_column[c];
            ++num_vectors;
          }

          row_idx.push_back(delta_row | (advance ? kAdvanceFlag : 0));
        }
      }
    }

    // Count number of advance flags encoded.
    int encoded = 0;
    for (int idx : row_idx) {
      if (idx & kAdvanceFlag) {
        encoded += (idx & kDoubleIndexEncode) ? 2 : 1;
      }
    }

    if (options_.high_fidelity_16bit_encode()) {
      ABSL_CHECK_EQ(2 * encoded, flow_compressed_16.size());
    } else {
      ABSL_CHECK_EQ(2 * encoded, flow_compressed_8.size());
    }

    // Adjust column start by compressions.
    int curr_adjust = 0;
    for (int k = 0; k < domain_width; ++k) {
      curr_adjust -= compressions_per_column[k];
      col_starts[k + 1] += curr_adjust;
      ABSL_CHECK_LE(col_starts[k], col_starts[k + 1]);
    }

    ABSL_CHECK_EQ(row_idx.size(), col_starts.back());
    ABSL_CHECK_EQ(num_vectors, row_idx.size() + compressible);
  }

  // Delta compress col_starts.
  std::vector<uint8_t> col_start_delta(domain_width + 1, 0);
  col_start_delta[0] = col_starts[0];
  for (int k = 1; k < domain_width + 1; ++k) {
    const int delta = col_starts[k] - col_starts[k - 1];
    ABSL_CHECK_LT(delta, 256) << "Only up to 255 items per column supported.";
    col_start_delta[k] = delta;
  }

  VLOG(1) << "error: " << average_error / (num_vectors + 1)
          << " additions: " << num_vectors - motion_data.num_elements();
  const Homography& background_model = tracking_data.background_model();

  const float scale_x = 1.0f / tracking_data.domain_width();
  const float scale_y = 1.0f / tracking_data.domain_height();

  Homography homog_scale = HomographyAdapter::Embed(
      AffineAdapter::FromArgs(0, 0, scale_x, 0, 0, scale_y));

  Homography inv_homog_scale = HomographyAdapter::Embed(
      AffineAdapter::FromArgs(0, 0, 1.0f / scale_x, 0, 0, 1.0f / scale_y));

  // Might be just the identity if not set.
  const Homography background_model_scaled =
      ModelCompose3(homog_scale, background_model, inv_homog_scale);

  std::string background_model_string =
      absl::StrCat(EncodeToString(background_model.h_00()),
                   EncodeToString(background_model.h_01()),
                   EncodeToString(background_model.h_02()),
                   EncodeToString(background_model.h_10()),
                   EncodeToString(background_model.h_11()),
                   EncodeToString(background_model.h_12()),
                   EncodeToString(background_model.h_20()),
                   EncodeToString(background_model.h_21()));

  std::string* data = binary_data->mutable_data();
  data->clear();
  int32_t vector_size = options_.high_fidelity_16bit_encode()
                            ? flow_compressed_16.size()
                            : flow_compressed_8.size();
  int32_t row_idx_size = row_idx.size();

  absl::StrAppend(data, EncodeToString(frame_flags),
                  EncodeToString(domain_width), EncodeToString(domain_height),
                  EncodeToString(frame_aspect), background_model_string,
                  EncodeToString(scale), EncodeToString(num_vectors),
                  EncodeVectorToString(col_start_delta),
                  EncodeToString(row_idx_size), EncodeVectorToString(row_idx),
                  EncodeToString(vector_size),
                  (options_.high_fidelity_16bit_encode()
                       ? EncodeVectorToString(flow_compressed_16)
                       : EncodeVectorToString(flow_compressed_8)));
  VLOG(1) << "Binary data size: " << data->size() << " for " << num_vectors
          << " (" << vector_size << ")";
}

std::string PopSubstring(int len, absl::string_view* piece) {
  std::string result = std::string(piece->substr(0, len));
  piece->remove_prefix(len);
  return result;
}

void FlowPackager::DecodeTrackingData(const BinaryTrackingData& container_data,
                                      TrackingData* tracking_data) const {
  ABSL_CHECK(tracking_data != nullptr);

  absl::string_view data(container_data.data());
  int32_t frame_flags = 0;
  int32_t domain_width = 0;
  int32_t domain_height = 0;
  std::vector<float> background_model;
  int32_t scale = 0;
  int32_t num_vectors = 0;
  float frame_aspect = 0.0f;

  DecodeFromStringView(PopSubstring(4, &data), &frame_flags);
  DecodeFromStringView(PopSubstring(4, &data), &domain_width);
  DecodeFromStringView(PopSubstring(4, &data), &domain_height);
  DecodeFromStringView(PopSubstring(4, &data), &frame_aspect);

  ABSL_CHECK_LE(domain_width, 256);
  ABSL_CHECK_LE(domain_height, 256);

  DecodeVectorFromStringView(
      PopSubstring(4 * HomographyAdapter::NumParameters(), &data),
      &background_model);
  DecodeFromStringView(PopSubstring(4, &data), &scale);
  DecodeFromStringView(PopSubstring(4, &data), &num_vectors);

  tracking_data->set_frame_flags(frame_flags);
  tracking_data->set_domain_width(domain_width);
  tracking_data->set_domain_height(domain_height);
  tracking_data->set_frame_aspect(frame_aspect);
  *tracking_data->mutable_background_model() =
      HomographyAdapter::FromFloatPointer(&background_model[0], false);

  TrackingData::MotionData* motion_data = tracking_data->mutable_motion_data();
  motion_data->set_num_elements(num_vectors);

  const bool high_profile = frame_flags & TrackingData::FLAG_PROFILE_HIGH;
  const bool high_fidelity =
      frame_flags & TrackingData::FLAG_HIGH_FIDELITY_VECTORS;
  const float flow_denom = 1.0f / scale;

  std::vector<uint8_t> col_starts_delta;
  DecodeVectorFromStringView(PopSubstring(domain_width + 1, &data),
                             &col_starts_delta);

  // Delta decompress.
  std::vector<int> col_starts;
  col_starts.reserve(domain_width + 1);

  int column = 0;
  for (auto col : col_starts_delta) {
    column += col;
    col_starts.push_back(column);
  }

  std::vector<uint8_t> row_idx;
  int32_t row_idx_size;
  DecodeFromStringView(PopSubstring(4, &data), &row_idx_size);

  // Should not have more row indices than vectors. (One for each in baseline
  // profile, less in high profile).
  ABSL_CHECK_LE(row_idx_size, num_vectors);
  DecodeVectorFromStringView(PopSubstring(row_idx_size, &data), &row_idx);

  // Records for each vector whether to advance pointer in the vector data array
  // or re-use previously read data.
  std::vector<bool> advance(num_vectors, true);

  if (high_profile) {
    // Unpack row indices, populate advance.
    const int kAdvanceFlag = FlowPackagerOptions::ADVANCE_FLAG;
    const int kDoubleIndexEncode = FlowPackagerOptions::DOUBLE_INDEX_ENCODE;
    const int kIndexMask = FlowPackagerOptions::INDEX_MASK;

    std::vector<int> column_expansions(domain_width, 0);
    std::vector<uint8_t> row_idx_unpacked;
    row_idx_unpacked.reserve(num_vectors);
    advance.clear();

    for (int c = 0; c < col_starts.size() - 1; ++c) {
      const int r_start = col_starts[c];
      const int r_end = col_starts[c + 1];
      uint8_t prev_row_idx = 0;
      for (int r = r_start; r < r_end; ++r) {
        // Use top bit as indicator to advance.
        advance.push_back(row_idx[r] & kAdvanceFlag);

        // Double encode?
        if (row_idx[r] & kDoubleIndexEncode) {
          // Indices are encoded as each 3 bit offset within kIndexMask.
          prev_row_idx += (row_idx[r] >> 3) & 0x7;
          row_idx_unpacked.push_back(prev_row_idx);
          prev_row_idx += row_idx[r] & 0x7;
          row_idx_unpacked.push_back(prev_row_idx);

          // Duplicate advance setting.
          advance.push_back(advance.back());
          ++column_expansions[c];
        } else {
          // Single encode.
          prev_row_idx += row_idx[r] & kIndexMask;  // Clear status.
          row_idx_unpacked.push_back(prev_row_idx);
        }
      }
    }
    row_idx.swap(row_idx_unpacked);
    ABSL_CHECK_EQ(num_vectors, row_idx.size());

    // Adjust column start by expansions.
    int curr_adjust = 0;
    for (int k = 0; k < domain_width; ++k) {
      curr_adjust += column_expansions[k];
      col_starts[k + 1] += curr_adjust;
    }
  }

  ABSL_CHECK_EQ(num_vectors, col_starts.back());

  int vector_data_size;
  DecodeFromStringView(PopSubstring(4, &data), &vector_data_size);

  int prev_flow_x = 0;
  int prev_flow_y = 0;
  if (high_fidelity) {
    std::vector<int16_t> vector_data;
    DecodeVectorFromStringView(
        PopSubstring(sizeof(vector_data[0]) * vector_data_size, &data),
        &vector_data);
    int counter = 0;
    for (int k = 0; k < num_vectors; ++k) {
      if (advance[k]) {  // Read new vector data.
        int flow_x = vector_data[counter++];
        int flow_y = vector_data[counter++];

        if (high_profile) {  // Delta decode in high profile.
          flow_x += prev_flow_x;
          flow_y += prev_flow_y;
          prev_flow_x = flow_x;
          prev_flow_y = flow_y;
        }

        motion_data->add_vector_data(flow_x * flow_denom);
        motion_data->add_vector_data(flow_y * flow_denom);
      } else {  // Re-use previous vector data.
        motion_data->add_vector_data(prev_flow_x * flow_denom);
        motion_data->add_vector_data(prev_flow_y * flow_denom);
      }
    }
    ABSL_CHECK_EQ(vector_data_size, counter);
  } else {
    std::vector<int8_t> vector_data;
    DecodeVectorFromStringView(
        PopSubstring(sizeof(vector_data[0]) * vector_data_size, &data),
        &vector_data);
    int counter = 0;
    for (int k = 0; k < num_vectors; ++k) {
      if (advance[k]) {  // Read new vector data.
        int flow_x = vector_data[counter++];
        int flow_y = vector_data[counter++];

        if (high_profile) {  // Delta decode in high profile.
          flow_x += prev_flow_x;
          flow_y += prev_flow_y;
          prev_flow_x = flow_x;
          prev_flow_y = flow_y;
        }

        motion_data->add_vector_data(flow_x * flow_denom);
        motion_data->add_vector_data(flow_y * flow_denom);
      } else {  // Re-use previous vector data.
        motion_data->add_vector_data(prev_flow_x * flow_denom);
        motion_data->add_vector_data(prev_flow_y * flow_denom);
      }
    }
    ABSL_CHECK_EQ(vector_data_size, counter);
  }

  for (auto idx : row_idx) {
    motion_data->add_row_indices(idx);
  }

  for (auto column : col_starts) {
    motion_data->add_col_starts(column);
  }
}

void FlowPackager::BinaryTrackingDataToContainer(
    const BinaryTrackingData& binary_data, TrackingContainer* container) const {
  ABSL_CHECK(container != nullptr);
  container->Clear();
  container->set_header("TRAK");
  container->set_version(1);
  container->set_size(binary_data.data().size());
  *container->mutable_data() = binary_data.data();
}

void FlowPackager::BinaryTrackingDataFromContainer(
    const TrackingContainer& container, BinaryTrackingData* binary_data) const {
  ABSL_CHECK_EQ("TRAK", container.header());
  ABSL_CHECK_EQ(1, container.version()) << "Unsupported version.";
  *binary_data->mutable_data() = container.data();
}

void FlowPackager::DecodeMetaData(const TrackingContainer& container_data,
                                  MetaData* meta_data) const {
  ABSL_CHECK(meta_data != nullptr);

  ABSL_CHECK_EQ("META", container_data.header());
  ABSL_CHECK_EQ(1, container_data.version()) << "Unsupported version.";

  absl::string_view data(container_data.data());

  int32_t num_frames;
  DecodeFromStringView(PopSubstring(4, &data), &num_frames);
  meta_data->set_num_frames(num_frames);

  for (int k = 0; k < num_frames; ++k) {
    int32_t msec;
    int32_t stream_offset;

    DecodeFromStringView(PopSubstring(4, &data), &msec);
    DecodeFromStringView(PopSubstring(4, &data), &stream_offset);

    MetaData::TrackOffset* track_offset = meta_data->add_track_offsets();
    track_offset->set_msec(msec);
    track_offset->set_stream_offset(stream_offset);
  }
}

void FlowPackager::FinalizeTrackingContainerFormat(
    std::vector<uint32_t>* timestamps,
    TrackingContainerFormat* container_format) {
  ABSL_CHECK(container_format != nullptr);

  // Compute binary sizes of track_data.
  const int num_frames = container_format->track_data_size();

  std::vector<uint32_t> msecs(num_frames, 0);
  if (timestamps) {
    ABSL_CHECK_EQ(num_frames, timestamps->size());
    msecs = *timestamps;
  }
  std::vector<int> sizes(num_frames, 0);

  for (int f = 0; f < num_frames; ++f) {
    // Default size of container: 12 bytes + binary data size (see comment for
    // TrackingContainer in flow_packager.proto).
    sizes[f] = container_format->track_data(f).data().size() + 12;
  }

  // Store relative offsets w.r.t. end of MetaData.
  MetaData meta_data;
  InitializeMetaData(num_frames, msecs, sizes, &meta_data);

  // Serialize metadata to binary.
  TrackingContainer* meta = container_format->mutable_meta_data();
  meta->Clear();
  meta->set_header("META");

  std::string* binary_metadata = meta->mutable_data();
  absl::StrAppend(binary_metadata, EncodeToString(meta_data.num_frames()));
  for (auto& track_offset : *meta_data.mutable_track_offsets()) {
    absl::StrAppend(binary_metadata, EncodeToString(track_offset.msec()),
                    EncodeToString(track_offset.stream_offset()));
  }

  meta->set_size(binary_metadata->size());

  // Add term header.
  TrackingContainer* term = container_format->mutable_term_data();
  term->set_header("TERM");
  term->set_size(0);
}

void FlowPackager::FinalizeTrackingContainerProto(
    std::vector<uint32_t>* timestamps, TrackingContainerProto* proto) {
  ABSL_CHECK(proto != nullptr);

  // Compute binary sizes of track_data.
  const int num_frames = proto->track_data_size();

  std::vector<uint32_t> msecs(num_frames, 0);
  if (timestamps) {
    ABSL_CHECK_EQ(num_frames, timestamps->size());
    msecs = *timestamps;
  }

  std::vector<int> sizes(num_frames, 0);

  TrackingContainerProto temp_proto;
  BinaryTrackingData* temp_track_data = temp_proto.add_track_data();
  for (int f = 0; f < num_frames; ++f) {
    // Swap current track data in and out of temp_track_data to determine total
    // encoding size with proto preamble.
    proto->mutable_track_data(f)->Swap(temp_track_data);
    sizes[f] = temp_proto.ByteSize();
    proto->mutable_track_data(f)->Swap(temp_track_data);
  }

  proto->clear_meta_data();
  InitializeMetaData(num_frames, msecs, sizes, proto->mutable_meta_data());
}

void FlowPackager::InitializeMetaData(int num_frames,
                                      const std::vector<uint32_t>& msecs,
                                      const std::vector<int>& data_sizes,
                                      MetaData* meta_data) const {
  meta_data->set_num_frames(num_frames);
  ABSL_CHECK_EQ(num_frames, msecs.size());
  ABSL_CHECK_EQ(num_frames, data_sizes.size());

  int curr_offset = 0;
  for (int f = 0; f < num_frames; ++f) {
    MetaData::TrackOffset* track_offset = meta_data->add_track_offsets();
    track_offset->set_msec(msecs[f]);
    track_offset->set_stream_offset(curr_offset);
    curr_offset += data_sizes[f];
  }
}

void FlowPackager::AddContainerToString(const TrackingContainer& container,
                                        std::string* binary_data) {
  ABSL_CHECK(binary_data != nullptr);
  std::string header_string(container.header());
  ABSL_CHECK_EQ(4, header_string.size());

  std::vector<char> header{header_string[0], header_string[1], header_string[2],
                           header_string[3]};
  absl::StrAppend(binary_data, EncodeVectorToString(header),
                  EncodeToString(container.version()),
                  EncodeToString(container.size()), container.data());
}

std::string FlowPackager::SplitContainerFromString(
    absl::string_view* binary_data, TrackingContainer* container) {
  ABSL_CHECK(binary_data != nullptr);
  ABSL_CHECK(container != nullptr);
  ABSL_CHECK_GE(binary_data->size(), 12) << "Data does not contain "
                                         << "valid container";

  container->set_header(PopSubstring(4, binary_data));

  int version;
  DecodeFromStringView(PopSubstring(4, binary_data), &version);

  int size;
  DecodeFromStringView(PopSubstring(4, binary_data), &size);

  container->set_version(version);
  container->set_size(size);

  if (size > 0) {
    container->set_data(PopSubstring(size, binary_data));
  }

  return container->header();
}

void FlowPackager::TrackingContainerFormatToBinary(
    const TrackingContainerFormat& container_format, std::string* binary) {
  ABSL_CHECK(binary != nullptr);
  binary->clear();

  AddContainerToString(container_format.meta_data(), binary);
  for (const auto& track_data : container_format.track_data()) {
    AddContainerToString(track_data, binary);
  }

  AddContainerToString(container_format.term_data(), binary);
}

void FlowPackager::TrackingContainerFormatFromBinary(
    const std::string& binary, TrackingContainerFormat* container_format) {
  ABSL_CHECK(container_format != nullptr);
  container_format->Clear();

  absl::string_view data(binary);

  ABSL_CHECK_EQ("META", SplitContainerFromString(
                            &data, container_format->mutable_meta_data()));
  MetaData meta_data;
  DecodeMetaData(container_format->meta_data(), &meta_data);

  for (int f = 0; f < meta_data.num_frames(); ++f) {
    TrackingContainer* container = container_format->add_track_data();
    ABSL_CHECK_EQ("TRAK", SplitContainerFromString(&data, container));
  }

  ABSL_CHECK_EQ("TERM", SplitContainerFromString(
                            &data, container_format->mutable_term_data()));
}

void FlowPackager::SortRegionFlowFeatureList(
    float scale_x, float scale_y, RegionFlowFeatureList* feature_list) const {
  ABSL_CHECK(feature_list != nullptr);
  // Sort features lexicographically.
  std::sort(feature_list->mutable_feature()->begin(),
            feature_list->mutable_feature()->end(),
            IntegerColumnComparator(scale_x, scale_y, options_.domain_width(),
                                    options_.domain_height()));
}

bool FlowPackager::CompatibleForEncodeWithoutDuplication(
    const TrackingData& tracking_data) const {
  const TrackingData::MotionData& motion_data = tracking_data.motion_data();
  for (int c = 0; c < motion_data.col_starts_size() - 1; ++c) {
    const int r_start = motion_data.col_starts(c);
    const int r_end = motion_data.col_starts(c + 1);
    for (int r = r_start; r < r_end; ++r) {
      if (motion_data.row_indices(r) -
              (r == r_start ? 0 : motion_data.row_indices(r - 1)) >=
          64) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace mediapipe
