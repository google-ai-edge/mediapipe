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

#include "mediapipe/util/tracking/tone_estimation.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "mediapipe/util/tracking/motion_models.pb.h"
#include "mediapipe/util/tracking/tone_models.pb.h"

namespace mediapipe {

ToneEstimation::ToneEstimation(const ToneEstimationOptions& options,
                               int frame_width, int frame_height)
    : options_(options),
      frame_width_(frame_width),
      frame_height_(frame_height),
      original_width_(frame_width),
      original_height_(frame_height) {
  switch (options_.downsample_mode()) {
    case ToneEstimationOptions::DOWNSAMPLE_NONE:
      break;
    case ToneEstimationOptions::DOWNSAMPLE_TO_MAX_SIZE: {
      const float max_size = std::max(frame_width_, frame_height_);
      if (max_size > 1.03f * options_.downsampling_size()) {
        downsample_scale_ = max_size / options_.downsampling_size();
        frame_height_ /= downsample_scale_;
        frame_width_ /= downsample_scale_;
        use_downsampling_ = true;
      }
      break;
    }
    case ToneEstimationOptions::DOWNSAMPLE_TO_MIN_SIZE: {
      const float min_size = std::min(frame_width_, frame_height_);
      if (min_size > 1.03f * options_.downsampling_size()) {
        downsample_scale_ = min_size / options_.downsampling_size();
        frame_height_ /= downsample_scale_;
        frame_width_ /= downsample_scale_;
        use_downsampling_ = true;
      }

      break;
    }
    case ToneEstimationOptions::DOWNSAMPLE_BY_FACTOR: {
      CHECK_GE(options_.downsample_factor(), 1);
      frame_width_ /= options_.downsample_factor();
      frame_height_ /= options_.downsample_factor();
      downsample_scale_ = options_.downsample_factor();
      use_downsampling_ = true;
      break;
    }
  }

  if (use_downsampling_) {
    resized_input_.reset(new cv::Mat(frame_height_, frame_width_, CV_8UC3));
    prev_resized_input_.reset(
        new cv::Mat(frame_height_, frame_width_, CV_8UC3));
  }
}

ToneEstimation::~ToneEstimation() {}

void ToneEstimation::EstimateToneChange(
    const RegionFlowFeatureList& feature_list_input,
    const cv::Mat& curr_frame_input, const cv::Mat* prev_frame_input,
    ToneChange* tone_change, cv::Mat* debug_output) {
  CHECK_EQ(original_height_, curr_frame_input.rows);
  CHECK_EQ(original_width_, curr_frame_input.cols);
  CHECK(tone_change != nullptr);

  const cv::Mat& curr_frame =
      use_downsampling_ ? *resized_input_ : curr_frame_input;
  const cv::Mat* prev_frame = (use_downsampling_ && prev_frame_input)
                                  ? prev_resized_input_.get()
                                  : prev_frame_input;

  RegionFlowFeatureList scaled_feature_list;
  const RegionFlowFeatureList& feature_list =
      use_downsampling_ ? scaled_feature_list : feature_list_input;

  if (use_downsampling_) {
    cv::resize(curr_frame_input, *resized_input_, resized_input_->size());
    if (prev_frame_input) {
      cv::resize(*prev_frame_input, *prev_resized_input_,
                 prev_resized_input_->size());
    }
    LinearSimilarityModel scale_transform;
    scale_transform.set_a(1.0f / downsample_scale_);
    scaled_feature_list = feature_list_input;
    TransformRegionFlowFeatureList(scale_transform, &scaled_feature_list);
  }

  CHECK_EQ(frame_height_, curr_frame.rows);
  CHECK_EQ(frame_width_, curr_frame.cols);

  ClipMask<3> curr_clip;
  ComputeClipMask<3>(options_.clip_mask_options(), curr_frame, &curr_clip);

  // Compute tone statistics.
  tone_change->set_frac_clipped(cv::sum(curr_clip.mask)[0] /
                                (frame_height_ * frame_width_));

  IntensityPercentiles(curr_frame, curr_clip.mask,
                       options_.tone_match_options().log_domain(), tone_change);

  ColorToneMatches color_tone_matches;
  // TODO: Buffer clip mask.
  if (prev_frame) {
    ClipMask<3> prev_clip;
    ComputeClipMask<3>(options_.clip_mask_options(), *prev_frame, &prev_clip);
    ComputeToneMatches<3>(options_.tone_match_options(), feature_list,
                          curr_frame, *prev_frame, curr_clip, prev_clip,
                          &color_tone_matches, debug_output);

    EstimateGainBiasModel(options_.irls_iterations(), &color_tone_matches,
                          tone_change->mutable_gain_bias());

    if (!IsStableGainBiasModel(options_.stable_gain_bias_bounds(),
                               tone_change->gain_bias(), color_tone_matches,
                               tone_change->mutable_stability_stats())) {
      VLOG(1) << "Warning: Estimated gain-bias is unstable.";
      // Reset to identity.
      tone_change->mutable_gain_bias()->CopyFrom(GainBiasModel());
      tone_change->set_type(ToneChange::INVALID);
    }

    // TODO: EstimateMixtureGainBiasModel();
  }
}

void ToneEstimation::IntensityPercentiles(const cv::Mat& frame,
                                          const cv::Mat& clip_mask,
                                          bool log_domain,
                                          ToneChange* tone_change) const {
  cv::Mat intensity(frame.rows, frame.cols, CV_8UC1);
  cv::cvtColor(frame, intensity, cv::COLOR_RGB2GRAY);

  std::vector<float> histogram(256, 0.0f);

  for (int i = 0; i < intensity.rows; ++i) {
    const uint8* intensity_ptr = intensity.ptr<uint8>(i);
    const uint8* clip_ptr = clip_mask.ptr<uint8>(i);

    for (int j = 0; j < intensity.cols; ++j) {
      if (!clip_ptr[j]) {
        ++histogram[intensity_ptr[j]];
      }
    }
  }

  // Construct cumulative histogram.
  std::partial_sum(histogram.begin(), histogram.end(), histogram.begin());

  // Normalize histogram.
  const float histogram_sum = histogram.back();
  if (histogram_sum == 0) {
    // Frame is of solid color. Use default values.
    return;
  }

  const float denom = 1.0f / histogram_sum;
  for (auto& entry : histogram) {
    entry *= denom;
  }

  std::vector<float> percentiles;
  percentiles.push_back(options_.stats_low_percentile());
  percentiles.push_back(options_.stats_low_mid_percentile());
  percentiles.push_back(options_.stats_mid_percentile());
  percentiles.push_back(options_.stats_high_mid_percentile());
  percentiles.push_back(options_.stats_high_percentile());

  std::vector<float> percentile_values(percentiles.size());

  const float log_denom = 1.0f / LogDomainLUT().MaxLogDomainValue();
  for (int k = 0; k < percentile_values.size(); ++k) {
    const int percentile_bin =
        std::lower_bound(histogram.begin(), histogram.end(), percentiles[k]) -
        histogram.begin();
    percentile_values[k] = percentile_bin;
    if (log_domain) {
      percentile_values[k] =
          LogDomainLUT().Map(percentile_values[k]) * log_denom;
    } else {
      percentile_values[k] *= (1.0f / 255.0f);
    }
  }

  tone_change->set_low_percentile(percentile_values[0]);
  tone_change->set_low_mid_percentile(percentile_values[1]);
  tone_change->set_mid_percentile(percentile_values[2]);
  tone_change->set_high_mid_percentile(percentile_values[3]);
  tone_change->set_high_percentile(percentile_values[4]);
}

void ToneEstimation::EstimateGainBiasModel(int irls_iterations,
                                           ColorToneMatches* color_tone_matches,
                                           GainBiasModel* gain_bias_model) {
  CHECK(color_tone_matches != nullptr);
  CHECK(gain_bias_model != nullptr);

  // Effectively estimate each model independently.
  float solution_ptr[6] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

  const int num_channels = color_tone_matches->size();
  CHECK_GT(num_channels, 0);
  CHECK_LE(num_channels, 3);

  // TODO: One IRLS weight per color match.
  for (int c = 0; c < num_channels; ++c) {
    std::deque<PatchToneMatch>& patch_tone_matches = (*color_tone_matches)[c];
    // Reset irls weight.
    int num_matches = 0;
    for (auto& patch_tone_match : patch_tone_matches) {
      patch_tone_match.set_irls_weight(1.0);
      num_matches += patch_tone_match.tone_match_size();
    }

    // Do not attempt solution if not matches have been found.
    if (num_matches < 3) {
      continue;
    }

    cv::Mat model_mat(num_matches, 2, CV_32F);
    cv::Mat rhs(num_matches, 1, CV_32F);
    cv::Mat solution(2, 1, CV_32F);

    for (int iteration = 0; iteration < irls_iterations; ++iteration) {
      // Setup matrix.
      int row = 0;
      for (const auto& patch_tone_match : patch_tone_matches) {
        const float irls_weight = patch_tone_match.irls_weight();
        for (const auto& tone_match : patch_tone_match.tone_match()) {
          float* row_ptr = model_mat.ptr<float>(row);
          float* rhs_ptr = rhs.ptr<float>(row);
          row_ptr[0] = tone_match.curr_val() * irls_weight;
          row_ptr[1] = irls_weight;
          rhs_ptr[0] = tone_match.prev_val() * irls_weight;
          ++row;
        }
      }

      // Solve.
      if (!cv::solve(model_mat, rhs, solution, cv::DECOMP_QR)) {
        // Fallback to identity.
        solution_ptr[2 * c] = 1;
        solution_ptr[2 * c + 1] = 0;
        break;  // Break to next color channel.
      }

      float a = solution.at<float>(0, 0);
      float b = solution.at<float>(1, 0);

      // Copy to solution.
      solution_ptr[2 * c] = a;
      solution_ptr[2 * c + 1] = b;

      // Evaluate error.
      for (auto& patch_tone_match : patch_tone_matches) {
        const int num_tone_matches = patch_tone_match.tone_match_size();

        if (num_tone_matches == 0) {
          continue;
        }

        float summed_error = 0.0f;
        for (const auto& tone_match : patch_tone_match.tone_match()) {
          // Express tone registration error in 0 .. 100.
          const float error =
              100.0f * (tone_match.curr_val() * a + b - tone_match.prev_val());
          summed_error += error * error;
        }

        // Compute RMSE.
        const float patch_error =
            std::sqrt(static_cast<double>(summed_error / num_tone_matches));
        // TODO: L1 instead of L0?
        patch_tone_match.set_irls_weight(1.0f / (patch_error + 1e-6f));
      }
    }
  }

  gain_bias_model->CopyFrom(
      GainBiasModelAdapter::FromPointer<float>(solution_ptr, false));

  // Test invertability, reset if failed.
  const float det = gain_bias_model->gain_c1() * gain_bias_model->gain_c2() *
                    gain_bias_model->gain_c3();
  if (fabs(det) < 1e-6f) {
    LOG(WARNING) << "Estimated gain bias model is not invertible. "
                 << "Falling back to identity model.";
    gain_bias_model->CopyFrom(GainBiasModel());
  }
}

bool ToneEstimation::IsStableGainBiasModel(
    const ToneEstimationOptions::GainBiasBounds& bounds,
    const GainBiasModel& model, const ColorToneMatches& color_tone_matches,
    ToneChange::StabilityStats* stats) {
  if (stats != nullptr) {
    stats->Clear();
  }

  // Test each channel for stability.
  if (model.gain_c1() < bounds.lower_gain() ||
      model.gain_c1() > bounds.upper_gain() ||
      model.bias_c1() < bounds.lower_bias() ||
      model.bias_c1() > bounds.upper_bias()) {
    return false;
  }

  if (model.gain_c2() < bounds.lower_gain() ||
      model.gain_c2() > bounds.upper_gain() ||
      model.bias_c2() < bounds.lower_bias() ||
      model.bias_c2() > bounds.upper_bias()) {
    return false;
  }

  if (model.gain_c3() < bounds.lower_gain() ||
      model.gain_c3() > bounds.upper_gain() ||
      model.bias_c3() < bounds.lower_bias() ||
      model.bias_c3() > bounds.upper_bias()) {
    return false;
  }

  // Test each channel independently.
  int total_inliers = 0;
  int total_tone_matches = 0;
  double total_inlier_weight = 0.0;
  for (const auto& patch_tone_matches : color_tone_matches) {
    int num_inliers = 0;
    for (const auto& patch_tone_match : patch_tone_matches) {
      if (patch_tone_match.irls_weight() > bounds.min_inlier_weight()) {
        ++num_inliers;
        // Clamp the weight to a registration error of 1 intensity value
        // difference (out of 255). Since weight are inversely proportional to
        // registration errors in the range 0..100, this corresponds to a max
        // weight of 2.55.
        total_inlier_weight += std::min(2.55f, patch_tone_match.irls_weight());
      }
    }

    if (num_inliers <
        bounds.min_inlier_fraction() * patch_tone_matches.size()) {
      return false;
    }

    total_inliers += num_inliers;
    total_tone_matches += patch_tone_matches.size();
  }

  if (stats != nullptr && total_tone_matches > 0) {
    stats->set_num_inliers(total_inliers);
    stats->set_inlier_fraction(total_inliers * 1.0f / total_tone_matches);
    stats->set_inlier_weight(total_inlier_weight);
  }

  return true;
}

}  // namespace mediapipe
