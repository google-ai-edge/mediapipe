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

#include "mediapipe/util/tracking/motion_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/tracking/camera_motion.h"
#include "mediapipe/util/tracking/camera_motion.pb.h"
#include "mediapipe/util/tracking/image_util.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/motion_saliency.pb.h"
#include "mediapipe/util/tracking/region_flow.h"
#include "mediapipe/util/tracking/region_flow.pb.h"
#include "mediapipe/util/tracking/region_flow_computation.h"
#include "mediapipe/util/tracking/region_flow_computation.pb.h"
#include "mediapipe/util/tracking/region_flow_visualization.h"

namespace mediapipe {

MotionAnalysis::MotionAnalysis(const MotionAnalysisOptions& options,
                               int frame_width, int frame_height)
    : options_(options),
      frame_width_(frame_width),
      frame_height_(frame_height) {
  // Init options by policy.
  InitPolicyOptions();
  // Merge back in any overriden options.
  options_.MergeFrom(options);

  region_flow_computation_.reset(new RegionFlowComputation(
      options_.flow_options(), frame_width_, frame_height_));
  motion_estimation_.reset(new MotionEstimation(options_.motion_options(),
                                                frame_width_, frame_height_));

  if (options_.compute_motion_saliency()) {
    motion_saliency_.reset(new MotionSaliency(options_.saliency_options(),
                                              frame_width_, frame_height_));

    // Compute overlap needed between clips if filtering or smoothing of
    // saliency is requested.
    if (options_.select_saliency_inliers()) {
      // 1.65 sigmas in each direction = 90% of variance is captured.
      overlap_size_ = std::max<int>(
          overlap_size_, options_.saliency_options().selection_frame_radius());
    }

    if (options_.filter_saliency()) {
      overlap_size_ = std::max<int>(
          overlap_size_,
          options_.saliency_options().filtering_sigma_time() * 1.65f);
    }
  }

  long_feature_stream_.reset(new LongFeatureStream);

  frame_num_ = 0;

  // Determine if feature descriptors need to be computed.
  // Required for irls smoothing, overlay detection and mixture homographies.
  const bool compute_mixtures =
      options_.motion_options().mix_homography_estimation() !=
      MotionEstimationOptions::ESTIMATION_HOMOG_MIX_NONE;

  const bool use_spatial_bias =
      options_.motion_options().estimation_policy() ==
          MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS &&
      options_.motion_options().long_feature_bias_options().use_spatial_bias();

  compute_feature_descriptors_ =
      options_.post_irls_smoothing() ||
      options_.motion_options().overlay_detection() || compute_mixtures ||
      use_spatial_bias;

  if (compute_feature_descriptors_) {
    CHECK_EQ(RegionFlowComputationOptions::FORMAT_RGB,
             options_.flow_options().image_format())
        << "Feature descriptors only support RGB currently.";
    prev_frame_.reset(new cv::Mat(frame_height_, frame_width_, CV_8UC3));
  }

  // Setup streaming buffer. By default we buffer features and motion.
  // If saliency is computed, also buffer saliency and filtered/smoothed
  // output_saliency.
  std::vector<TaggedType> data_config{
      TaggedPointerType<RegionFlowFeatureList>("features"),
      TaggedPointerType<CameraMotion>("motion")};
  std::vector<TaggedType> data_config_saliency = data_config;
  data_config_saliency.push_back(
      TaggedPointerType<SalientPointFrame>("saliency"));
  data_config_saliency.push_back(
      TaggedPointerType<SalientPointFrame>("output_saliency"));

  // Store twice the overlap.
  buffer_.reset(new StreamingBuffer(
      options_.compute_motion_saliency() ? data_config_saliency : data_config,
      2 * overlap_size_));
}

void MotionAnalysis::InitPolicyOptions() {
  auto* flow_options = options_.mutable_flow_options();
  auto* tracking_options = flow_options->mutable_tracking_options();
  auto* motion_options = options_.mutable_motion_options();
  auto* feature_bias_options =
      motion_options->mutable_long_feature_bias_options();
  auto* translation_bounds =
      motion_options->mutable_stable_translation_bounds();
  auto* similarity_bounds = motion_options->mutable_stable_similarity_bounds();
  auto* homography_bounds = motion_options->mutable_stable_homography_bounds();

  switch (options_.analysis_policy()) {
    case MotionAnalysisOptions::ANALYSIS_POLICY_LEGACY:
      break;

    case MotionAnalysisOptions::ANALYSIS_POLICY_VIDEO:
      // Long track settings. Temporally consistent.
      options_.set_estimation_clip_size(64);

      tracking_options->set_internal_tracking_direction(
          TrackingOptions::FORWARD);
      tracking_options->set_tracking_policy(
          TrackingOptions::POLICY_LONG_TRACKS);

      feature_bias_options->set_use_spatial_bias(false);
      feature_bias_options->set_seed_priors_from_bias(true);

      similarity_bounds->set_min_inlier_fraction(0.15f);
      homography_bounds->set_min_inlier_coverage(0.25f);
      flow_options->set_verify_long_features(false);

      // Better features.
      tracking_options->set_adaptive_features_levels(3);
      tracking_options->set_use_cv_tracking_algorithm(true);

      // Speed.
      flow_options->set_downsample_mode(
          RegionFlowComputationOptions::DOWNSAMPLE_BY_SCHEDULE);

      motion_options->set_estimation_policy(
          MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS);
      motion_options->set_feature_density_normalization(true);
      motion_options->set_overlay_detection(true);
      motion_options->set_domain_limited_irls_scaling(true);
      motion_options->set_irls_weights_preinitialized(true);
      motion_options->mutable_irls_initialization()->set_activated(true);
      motion_options->mutable_long_feature_initialization()->set_activated(
          true);

      break;

    case MotionAnalysisOptions::ANALYSIS_POLICY_VIDEO_MOBILE:
      // Long track settings. Temporally consistent.
      options_.set_estimation_clip_size(32);
      tracking_options->set_internal_tracking_direction(
          TrackingOptions::FORWARD);
      tracking_options->set_tracking_policy(
          TrackingOptions::POLICY_LONG_TRACKS);

      motion_options->set_estimation_policy(
          MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS);
      motion_options->set_feature_density_normalization(true);
      motion_options->set_domain_limited_irls_scaling(true);
      motion_options->mutable_irls_initialization()->set_activated(true);
      motion_options->mutable_long_feature_initialization()->set_activated(
          true);

      feature_bias_options->set_use_spatial_bias(false);
      feature_bias_options->set_seed_priors_from_bias(true);

      similarity_bounds->set_inlier_threshold(2.0);
      similarity_bounds->set_min_inlier_fraction(0.15f);

      // Speed for mobile.
      tracking_options->set_tracking_window_size(8);
      tracking_options->set_tracking_iterations(8);
      tracking_options->set_fractional_tracking_distance(0.1);
      tracking_options->set_reuse_features_max_frame_distance(15);
      tracking_options->set_max_features(500);
      tracking_options->set_use_cv_tracking_algorithm(true);

      flow_options->set_downsample_mode(
          RegionFlowComputationOptions::DOWNSAMPLE_TO_MIN_SIZE);
      flow_options->set_round_downsample_factor(true);
      flow_options->set_downsampling_size(256);
      flow_options->set_pre_blur_sigma(0);
      flow_options->set_top_inlier_sets(1);
      flow_options->set_ransac_rounds_per_region(10);
      flow_options->mutable_visual_consistency_options()
          ->set_compute_consistency(false);
      flow_options->set_verify_long_features(false);

      // Avoid resetting features if a large amount is flagged.
      // In that case verify feature tracks to avoid pertubing the estimation.
      flow_options->set_verify_long_feature_acceleration(true);
      flow_options->set_verification_distance(5.0);
      flow_options->set_verify_long_feature_trigger_ratio(0.1);

      motion_options->set_use_exact_homography_estimation(false);
      motion_options->set_use_highest_accuracy_for_normal_equations(false);

      break;

    case MotionAnalysisOptions::ANALYSIS_POLICY_CAMERA_MOBILE:
      // Long track settings.
      tracking_options->set_internal_tracking_direction(
          TrackingOptions::FORWARD);
      tracking_options->set_tracking_policy(
          TrackingOptions::POLICY_LONG_TRACKS);

      motion_options->set_estimation_policy(
          MotionEstimationOptions::TEMPORAL_IRLS_MASK);
      motion_options->set_feature_density_normalization(true);
      motion_options->set_domain_limited_irls_scaling(true);
      motion_options->mutable_irls_initialization()->set_activated(true);
      motion_options->mutable_irls_initialization()->set_rounds(50);

      feature_bias_options->set_use_spatial_bias(false);
      feature_bias_options->set_seed_priors_from_bias(true);

      similarity_bounds->set_inlier_threshold(2.0);
      similarity_bounds->set_min_inlier_fraction(0.15f);

      // Speed for mobile.
      tracking_options->set_tracking_window_size(8);
      tracking_options->set_tracking_iterations(8);
      tracking_options->set_fractional_tracking_distance(0.1);
      tracking_options->set_reuse_features_max_frame_distance(10);
      tracking_options->set_reuse_features_min_survived_frac(0.6);
      tracking_options->set_max_features(240);
      tracking_options->set_adaptive_tracking_distance(true);
      tracking_options->set_use_cv_tracking_algorithm(true);

      // Assumes downsampled input.
      flow_options->set_pre_blur_sigma(0);
      flow_options->set_top_inlier_sets(1);
      flow_options->set_ransac_rounds_per_region(10);
      flow_options->mutable_visual_consistency_options()
          ->set_compute_consistency(false);
      flow_options->set_verify_long_features(false);

      motion_options->set_use_exact_homography_estimation(false);
      motion_options->set_use_highest_accuracy_for_normal_equations(false);

      // Low latency.
      options_.set_estimation_clip_size(1);
      break;

    case MotionAnalysisOptions::ANALYSIS_POLICY_HYPERLAPSE:
      // Long track settings. Temporally consistent.
      options_.set_estimation_clip_size(64);

      tracking_options->set_internal_tracking_direction(
          TrackingOptions::FORWARD);
      tracking_options->set_tracking_policy(
          TrackingOptions::POLICY_LONG_TRACKS);

      feature_bias_options->set_use_spatial_bias(false);
      feature_bias_options->set_seed_priors_from_bias(true);

      // Relaxed stability bounds.
      translation_bounds->set_max_motion_stdev(0.15);
      translation_bounds->set_max_acceleration(60.0);
      translation_bounds->set_frac_max_motion_magnitude(0.25);

      similarity_bounds->set_min_inlier_fraction(0.02f);
      similarity_bounds->set_min_inliers(5);
      similarity_bounds->set_lower_scale(0.5);
      similarity_bounds->set_upper_scale(2.0);
      similarity_bounds->set_limit_rotation(0.5);

      homography_bounds->set_lower_scale(0.5);
      homography_bounds->set_upper_scale(2.0);
      homography_bounds->set_limit_rotation(0.5);
      homography_bounds->set_limit_perspective(0.004);
      homography_bounds->set_min_inlier_coverage(0.1);

      // Relax constraining of homographies.
      motion_options->set_lin_sim_inlier_threshold(0.02);
      // Anticipate much higher registration errors.
      motion_options->set_irls_motion_magnitude_fraction(0.2);

      flow_options->set_verify_long_features(false);

      // Better features.
      tracking_options->set_adaptive_features_levels(3);

      // Speed.
      tracking_options->set_use_cv_tracking_algorithm(true);
      flow_options->set_downsample_mode(
          RegionFlowComputationOptions::DOWNSAMPLE_BY_SCHEDULE);

      // Less quality features.
      flow_options->set_absolute_inlier_error_threshold(4);
      flow_options->set_relative_inlier_error_threshold(0.2);

      motion_options->set_estimation_policy(
          MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS);
      motion_options->set_feature_density_normalization(true);

      // TODO: This needs to be activated and then tighten relaxed
      // thresholds above again.
      //  motion_options->set_domain_limited_irls_scaling(true);

      motion_options->set_irls_weights_preinitialized(true);
      motion_options->mutable_irls_initialization()->set_activated(true);

      // More weight towards long feature tracks.
      motion_options->mutable_long_feature_initialization()->set_activated(
          true);
      motion_options->mutable_long_feature_initialization()
          ->set_min_length_percentile(0.975);
      motion_options->mutable_long_feature_initialization()
          ->set_upweight_multiplier(50);

      // 2nd pass, no need to do overlay again.
      motion_options->set_overlay_detection(false);

      break;
  }
}

bool MotionAnalysis::AddFrame(const cv::Mat& frame, int64 timestamp_usec,
                              RegionFlowFeatureList* feature_list) {
  return AddFrameWithSeed(frame, timestamp_usec, Homography(), feature_list);
}

bool MotionAnalysis::AddFrameWithSeed(const cv::Mat& frame,
                                      int64 timestamp_usec,
                                      const Homography& initial_transform,
                                      RegionFlowFeatureList* feature_list) {
  return AddFrameGeneric(frame, timestamp_usec, initial_transform,
                         nullptr,  // rejection_transform
                         nullptr,  // external features
                         nullptr,  // feature modification function
                         feature_list);
}

bool MotionAnalysis::AddFrameGeneric(
    const cv::Mat& frame, int64 timestamp_usec,
    const Homography& initial_transform, const Homography* rejection_transform,
    const RegionFlowFeatureList* external_features,
    std::function<void(RegionFlowFeatureList*)>* modify_features,
    RegionFlowFeatureList* output_feature_list) {
  // Don't check input sizes here, RegionFlowComputation does that based
  // on its internal options.
  CHECK(feature_computation_) << "Calls to AddFrame* can NOT be mixed "
                              << "with AddFeatures";

  // Compute RegionFlow.
  {
    MEASURE_TIME << "CALL RegionFlowComputation::AddImage";
    if (!region_flow_computation_->AddImageWithSeed(frame, timestamp_usec,
                                                    initial_transform)) {
      LOG(ERROR) << "Error while computing region flow.";
      return false;
    }
  }

  // Buffer features.
  std::unique_ptr<RegionFlowFeatureList> feature_list;
  {
    MEASURE_TIME << "CALL RegionFlowComputation::RetrieveRegionFlowFeatureList";
    const bool compute_feature_match_descriptors =
        compute_feature_descriptors_ && frame_num_ > 0;

    // Determine which region flow should be computed in case multi frame
    // tracking has been requested.
    int max_track_index = 0;
    if (options_.flow_options().tracking_options().tracking_policy() ==
        TrackingOptions::POLICY_MULTI_FRAME) {
      max_track_index = std::min(
          options_.flow_options().tracking_options().multi_frames_to_track() -
              1,
          options_.track_index());
    }

    feature_list.reset(
        region_flow_computation_->RetrieveMultiRegionFlowFeatureList(
            std::min(std::max(0, frame_num_ - 1), max_track_index),
            compute_feature_descriptors_, compute_feature_match_descriptors,
            &frame,
            compute_feature_match_descriptors ? prev_frame_.get() : nullptr));

    if (feature_list == nullptr) {
      LOG(ERROR) << "Error retrieving feature list.";
      return false;
    }
  }

  if (external_features) {
    constexpr int kTrackIdShift = 1 << 20;
    for (const auto& external_feat : external_features->feature()) {
      auto* new_feat = feature_list->add_feature();
      *new_feat = external_feat;
      if (new_feat->track_id() >= 0) {
        new_feat->set_track_id(new_feat->track_id() + kTrackIdShift);
      }
    }
  }

  if (rejection_transform) {
    RegionFlowFeatureList tmp_list;
    tmp_list.mutable_feature()->Swap(feature_list->mutable_feature());

    for (const auto& feature : tmp_list.feature()) {
      const Vector2_f diff =
          TransformPoint(*rejection_transform, FeatureLocation(feature)) -
          FeatureMatchLocation(feature);
      if (diff.Norm() < options_.rejection_transform_threshold()) {
        *feature_list->add_feature() = feature;
      }
    }
  }

  if (output_feature_list) {
    *output_feature_list = *feature_list;
  }

  if (modify_features) {
    (*modify_features)(feature_list.get());
  }

  buffer_->EmplaceDatum("features", feature_list.release());

  // Store frame for next call.
  if (compute_feature_descriptors_) {
    frame.copyTo(*prev_frame_);
  }

  ++frame_num_;

  return true;
}

void MotionAnalysis::AddFeatures(const RegionFlowFeatureList& features) {
  feature_computation_ = false;
  buffer_->EmplaceDatum("features", new RegionFlowFeatureList(features));

  ++frame_num_;
}

void MotionAnalysis::EnqueueFeaturesAndMotions(
    const RegionFlowFeatureList& features, const CameraMotion& motion) {
  feature_computation_ = false;
  CHECK(buffer_->HaveEqualSize({"motion", "features"}))
      << "Can not be mixed with other Add* calls";
  buffer_->EmplaceDatum("features", new RegionFlowFeatureList(features));
  buffer_->EmplaceDatum("motion", new CameraMotion(motion));
}

cv::Mat MotionAnalysis::GetGrayscaleFrameFromResults() {
  return region_flow_computation_->GetGrayscaleFrameFromResults();
}

int MotionAnalysis::GetResults(
    bool flush, std::vector<std::unique_ptr<RegionFlowFeatureList>>* features,
    std::vector<std::unique_ptr<CameraMotion>>* camera_motion,
    std::vector<std::unique_ptr<SalientPointFrame>>* saliency) {
  MEASURE_TIME << "GetResults";

  const int num_features_lists = buffer_->BufferSize("features");
  const int num_new_feature_lists = num_features_lists - overlap_start_;
  CHECK_GE(num_new_feature_lists, 0);

  if (!flush && num_new_feature_lists < options_.estimation_clip_size()) {
    // Nothing to compute, return.
    return 0;
  }

  const bool compute_saliency = options_.compute_motion_saliency();
  CHECK_EQ(compute_saliency, saliency != nullptr)
      << "Computing saliency requires saliency output and vice versa";

  // Estimate motions for newly buffered RegionFlowFeatureLists, which also
  // computes IRLS feature weights for foreground estimation, if needed
  // (otherwise could be externally added).
  const int num_motions_to_compute =
      buffer_->BufferSize("features") - buffer_->BufferSize("motion");

  if (num_motions_to_compute > 0) {
    std::vector<CameraMotion> camera_motions;
    std::vector<RegionFlowFeatureList*> feature_lists;
    for (int k = overlap_start_; k < num_features_lists; ++k) {
      feature_lists.push_back(
          buffer_->GetMutableDatum<RegionFlowFeatureList>("features", k));
    }

    // TODO: Result should be vector of unique_ptr.
    motion_estimation_->EstimateMotionsParallel(
        options_.post_irls_smoothing(), &feature_lists, &camera_motions);

    // Add solution to buffer.
    for (const auto& motion : camera_motions) {
      buffer_->EmplaceDatum("motion", new CameraMotion(motion));
    }
  }

  CHECK(buffer_->HaveEqualSize({"features", "motion"}));

  if (compute_saliency) {
    ComputeSaliency();
  }

  return OutputResults(flush, features, camera_motion, saliency);
}

int MotionAnalysis::OutputResults(
    bool flush, std::vector<std::unique_ptr<RegionFlowFeatureList>>* features,
    std::vector<std::unique_ptr<CameraMotion>>* camera_motion,
    std::vector<std::unique_ptr<SalientPointFrame>>* saliency) {
  const bool compute_saliency = options_.compute_motion_saliency();
  CHECK_EQ(compute_saliency, saliency != nullptr)
      << "Computing saliency requires saliency output and vice versa";
  CHECK(buffer_->HaveEqualSize({"features", "motion"}));

  // Discard prev. overlap (already output, just used for filtering here).
  buffer_->DiscardData(buffer_->AllTags(), prev_overlap_start_);
  prev_overlap_start_ = 0;

  // Output only frames not part of the overlap.
  const int num_output_frames =
      std::max(0, buffer_->MaxBufferSize() - (flush ? 0 : overlap_size_));

  if (features) {
    features->reserve(num_output_frames);
  }
  if (camera_motion) {
    camera_motion->reserve(num_output_frames);
  }
  if (saliency) {
    saliency->reserve(num_output_frames);
  }

  // Need to retain twice the overlap, for frames in that segment we need
  // to create a copy.
  const int new_overlap_start =
      std::max(0, num_output_frames - (flush ? 0 : overlap_size_));

  // Populate output.
  for (int k = 0; k < num_output_frames; ++k) {
    std::unique_ptr<RegionFlowFeatureList> out_features;
    std::unique_ptr<CameraMotion> out_motion;
    std::unique_ptr<SalientPointFrame> out_saliency;

    if (k >= new_overlap_start) {
      // Create copy.
      out_features.reset(new RegionFlowFeatureList(
          *buffer_->GetDatum<RegionFlowFeatureList>("features", k)));
      out_motion.reset(
          new CameraMotion(*buffer_->GetDatum<CameraMotion>("motion", k)));
    } else {
      // Release datum.
      out_features =
          buffer_->ReleaseDatum<RegionFlowFeatureList>("features", k);
      out_motion = buffer_->ReleaseDatum<CameraMotion>("motion", k);
    }

    // output_saliency is temporary so we never need to buffer it.
    if (compute_saliency) {
      out_saliency =
          buffer_->ReleaseDatum<SalientPointFrame>("output_saliency", k);
    }

    if (options_.subtract_camera_motion_from_features()) {
      std::vector<RegionFlowFeatureList*> feature_view{out_features.get()};
      SubtractCameraMotionFromFeatures({*out_motion}, &feature_view);
    }

    if (features != nullptr) {
      features->push_back(std::move(out_features));
    }
    if (camera_motion != nullptr) {
      camera_motion->push_back(std::move(out_motion));
    }
    if (saliency != nullptr) {
      saliency->push_back(std::move(out_saliency));
    }
  }

  // Reset for next chunk.
  prev_overlap_start_ = num_output_frames - new_overlap_start;
  CHECK_GE(prev_overlap_start_, 0);

  CHECK(buffer_->TruncateBuffer(flush));

  overlap_start_ = buffer_->MaxBufferSize();
  return num_output_frames;
}

void MotionAnalysis::RenderResults(const RegionFlowFeatureList& feature_list,
                                   const CameraMotion& motion,
                                   const SalientPointFrame* saliency,
                                   cv::Mat* rendered_results) {
#ifndef NO_RENDERING
  CHECK(rendered_results != nullptr);
  CHECK_EQ(frame_width_, rendered_results->cols);
  CHECK_EQ(frame_height_, rendered_results->rows);

  const auto viz_options = options_.visualization_options();

  // Visualize flow features if requested.
  if (viz_options.visualize_region_flow_features()) {
    cv::Scalar inlier_color(0, 255, 0);
    cv::Scalar outlier_color(255, 0, 0);
    if (feature_list.long_tracks()) {
      long_feature_stream_->AddFeatures(feature_list,
                                        true,   // Check connectivity.
                                        true);  // Purge non present ones.

      VisualizeLongFeatureStream(
          *long_feature_stream_, inlier_color, outlier_color,
          viz_options.min_long_feature_track(),
          viz_options.max_long_feature_points(), 1.0f, 1.0f, rendered_results);
    } else {
      VisualizeRegionFlowFeatures(feature_list, inlier_color, outlier_color,
                                  true, 1.0f, 1.0f, rendered_results);
    }
  }

  if (saliency != nullptr && viz_options.visualize_salient_points()) {
    static const cv::Scalar kColor(255, 0, 0);
    RenderSaliency(*saliency, kColor, viz_options.line_thickness(), false,
                   rendered_results);
  }

  if (viz_options.visualize_blur_analysis_region()) {
    VisualizeBlurAnalysisRegions(rendered_results);
  }

  if (viz_options.visualize_stats()) {
    // Output general stats.
    std::string hud_text = absl::StrFormat(
        "H-cvg %.2f | H-err %4.2f | Avg.t %3.1f | dx: %+2.1f dy: %+2.1f "
        "Feat# %4d | %s | ",
        motion.homography_inlier_coverage(), motion.average_homography_error(),
        motion.average_magnitude(), motion.translation().dx(),
        motion.translation().dy(), feature_list.feature_size(),
        CameraMotionTypeToString(motion));
    hud_text += CameraMotionFlagToString(motion);

    const float text_scale = frame_width_ * 5.e-4;
    cv::putText(*rendered_results, hud_text,
                cv::Point(0.02 * frame_width_, 0.975 * frame_height_),
                cv::FONT_HERSHEY_SIMPLEX, text_scale, cv::Scalar::all(255),
                text_scale * 3, cv::LINE_AA);

    cv::putText(*rendered_results,
                absl::StrFormat("%6d", motion.timestamp_usec() / 1000),
                cv::Point(0.9 * frame_width_, 0.05 * frame_height_),
                cv::FONT_HERSHEY_SIMPLEX, text_scale, cv::Scalar::all(255),
                text_scale * 3, cv::LINE_AA);
  }
#else
  LOG(FATAL) << "Code stripped out because of NO_RENDERING";
#endif
}

void MotionAnalysis::ComputeDenseForeground(
    const RegionFlowFeatureList& feature_list,
    const CameraMotion& camera_motion, cv::Mat* foreground_mask) {
  const auto& foreground_options = options_.foreground_options();

  if (foreground_push_pull_ == nullptr) {
    foreground_push_pull_.reset(
        new PushPullFilteringC1(cv::Size(frame_width_, frame_height_),
                                PushPullFilteringC1::BINOMIAL_5X5, false,
                                nullptr,    // Gaussian filter weights only.
                                nullptr,    // No mip map visualizer.
                                nullptr));  // No weight adjustment.
  }

  // Determine foreground weights for each features.
  std::vector<float> foreground_weights;
  ForegroundWeightsFromFeatures(
      feature_list, foreground_options.foreground_threshold(),
      foreground_options.foreground_gamma(),
      foreground_options.threshold_coverage_scaling() ? &camera_motion
                                                      : nullptr,
      &foreground_weights);

  // Setup push pull map (with border). Ensure constructor used the right type.
  CHECK(foreground_push_pull_->filter_type() ==
            PushPullFilteringC1::BINOMIAL_5X5 ||
        foreground_push_pull_->filter_type() ==
            PushPullFilteringC1::GAUSSIAN_5X5);

  cv::Mat foreground_map(frame_height_ + 4, frame_width_ + 4, CV_32FC2);
  std::vector<Vector2_f> feature_locations;
  std::vector<cv::Vec<float, 1>> feature_irls;

  for (int feat_idx = 0; feat_idx < foreground_weights.size(); ++feat_idx) {
    // Skip marked outliers.
    if (foreground_weights[feat_idx] == 0) {
      continue;
    }

    feature_locations.push_back(
        FeatureLocation(feature_list.feature(feat_idx)));
    feature_irls.push_back(cv::Vec<float, 1>(foreground_weights[feat_idx]));
  }

  // Run push pull.
  foreground_push_pull_->PerformPushPull(feature_locations, feature_irls, 0.2,
                                         cv::Point2i(0, 0),
                                         0,        // Default read out level.
                                         nullptr,  // Uniform weights.
                                         nullptr,  // No bilateral term.
                                         &foreground_map);

  // Convert to grayscale output.
  foreground_mask->create(frame_height_, frame_width_, CV_8U);
  for (int i = 0; i < frame_height_; ++i) {
    const float* src_ptr = foreground_map.ptr<float>(i);
    uint8* dst_ptr = foreground_mask->ptr<uint8>(i);
    for (int j = 0; j < frame_width_; ++j) {
      // Result is in first channel (second is confidence).
      dst_ptr[j] =
          std::max(0, std::min(255, static_cast<int>(src_ptr[2 * j] * 255.0f)));
    }
  }
}

void MotionAnalysis::VisualizeDenseForeground(const cv::Mat& foreground_mask,
                                              cv::Mat* output) {
  CHECK(output != nullptr);
  CHECK(foreground_mask.size() == output->size());
  // Map foreground measure to color (green by default).
  std::vector<Vector3_f> color_map;
  if (options_.visualization_options().foreground_jet_coloring()) {
    JetColoring(1000, &color_map);
  } else {
    color_map.resize(1000, Vector3_f(0, 255, 0));
  }

  auto clamp = [](int value) -> int {
    return std::max(0, std::min(255, value));
  };

  // Burn-in alpha compositing.
  const float alpha = 1.3f;
  for (int i = 0; i < frame_height_; ++i) {
    uint8* image_ptr = output->ptr<uint8>(i);
    const uint8* foreground_ptr = foreground_mask.ptr<uint8>(i);

    for (int j = 0; j < frame_width_; ++j) {
      const float norm_foreground = foreground_ptr[j] * (1.0 / 255.0f);
      const float foreground = norm_foreground * alpha;
      const float alpha_denom = 1.0f / (1.0f + foreground);

      Vector3_f color =
          color_map[static_cast<int>(norm_foreground * (color_map.size() - 1))];

      image_ptr[3 * j] =
          clamp((image_ptr[3 * j] + color[0] * foreground) * alpha_denom);
      image_ptr[3 * j + 1] =
          clamp((image_ptr[3 * j + 1] + color[1] * foreground) * alpha_denom);
      image_ptr[3 * j + 2] =
          clamp((image_ptr[3 * j + 2] + color[2] * foreground) * alpha_denom);
    }
  }
}

void MotionAnalysis::VisualizeBlurAnalysisRegions(cv::Mat* input_view) {
  CHECK(input_view != nullptr);

  cv::Mat intensity;
  cv::cvtColor(*input_view, intensity, cv::COLOR_RGB2GRAY);
  cv::Mat corner_values;
  cv::cornerMinEigenVal(intensity, corner_values, 3);

  cv::Mat mask;
  region_flow_computation_->ComputeBlurMask(*input_view, &corner_values, &mask);

  cv::Mat mask_3c;
  cv::cvtColor(mask, mask_3c, CV_GRAY2RGB);
  cv::addWeighted(*input_view, 0.5, mask_3c, 0.5, -128, *input_view);
}

void MotionAnalysis::ComputeSaliency() {
  MEASURE_TIME << "Saliency computation.";
  CHECK_EQ(overlap_start_, buffer_->BufferSize("saliency"));

  const int num_features_lists = buffer_->BufferSize("features");

  // Compute saliency only for newly buffered RegionFlowFeatureLists.
  for (int k = overlap_start_; k < num_features_lists; ++k) {
    std::vector<float> foreground_weights;
    ForegroundWeightsFromFeatures(
        *buffer_->GetDatum<RegionFlowFeatureList>("features", k),
        options_.foreground_options().foreground_threshold(),
        options_.foreground_options().foreground_gamma(),
        options_.foreground_options().threshold_coverage_scaling()
            ? buffer_->GetDatum<CameraMotion>("motion", k)
            : nullptr,
        &foreground_weights);

    std::unique_ptr<SalientPointFrame> saliency(new SalientPointFrame());
    motion_saliency_->SaliencyFromFeatures(
        *buffer_->GetDatum<RegionFlowFeatureList>("features", k),
        &foreground_weights, saliency.get());

    buffer_->AddDatum("saliency", std::move(saliency));
  }

  CHECK(buffer_->HaveEqualSize({"features", "motion", "saliency"}));

  // Clear output saliency and copy from saliency.
  buffer_->DiscardDatum("output_saliency",
                        buffer_->BufferSize("output_saliency"));

  for (int k = 0; k < buffer_->BufferSize("saliency"); ++k) {
    std::unique_ptr<SalientPointFrame> copy(new SalientPointFrame());
    *copy = *buffer_->GetDatum<SalientPointFrame>("saliency", k);
    buffer_->AddDatum("output_saliency", std::move(copy));
  }

  // Create view.
  std::vector<SalientPointFrame*> saliency_view =
      buffer_->GetMutableDatumVector<SalientPointFrame>("output_saliency");

  // saliency_frames are filtered after this point and ready for output.
  if (options_.select_saliency_inliers()) {
    motion_saliency_->SelectSaliencyInliers(&saliency_view, false);
  }

  if (options_.filter_saliency()) {
    motion_saliency_->FilterMotionSaliency(&saliency_view);
  }
}

}  // namespace mediapipe
