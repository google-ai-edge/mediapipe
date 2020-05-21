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

#include "mediapipe/util/tracking/motion_estimation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/QR"
#include "Eigen/SVD"
#include "absl/container/node_hash_map.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/util/tracking/camera_motion.h"
#include "mediapipe/util/tracking/measure_time.h"
#include "mediapipe/util/tracking/motion_models.h"
#include "mediapipe/util/tracking/motion_models.pb.h"
#include "mediapipe/util/tracking/parallel_invoker.h"
#include "mediapipe/util/tracking/region_flow.h"
#include "mediapipe/util/tracking/region_flow.pb.h"

namespace mediapipe {

constexpr float kIrlsEps = 1e-4f;
constexpr float kOutlierIRLSWeight = 1e-10f;
constexpr float kMaxCondition = 1e30f;

constexpr float kPrecision = 0.1f;

typedef RegionFlowFrame::RegionFlow RegionFlow;
typedef RegionFlowFeature Feature;

namespace {

void GenericFit(
    const RegionFlowFeatureList& features,
    const std::function<bool(MotionEstimation*, RegionFlowFeatureList*,
                             CameraMotion*)>& est_func,
    CameraMotion* motion) {
  MotionEstimationOptions options;
  options.set_irls_rounds(1);
  options.set_use_exact_homography_estimation(false);
  options.set_use_highest_accuracy_for_normal_equations(false);

  MotionEstimation motion_est(options, features.frame_width(),
                              features.frame_height());

  RegionFlowFeatureList local = features;
  NormalizeRegionFlowFeatureList(&local);
  est_func(&motion_est, &local, motion);
}

}  // namespace.

TranslationModel FitTranslationModel(const RegionFlowFeatureList& features) {
  CameraMotion motion;
  GenericFit(features, &MotionEstimation::EstimateTranslationModel, &motion);
  return motion.translation();
}

LinearSimilarityModel FitLinearSimilarityModel(
    const RegionFlowFeatureList& features) {
  CameraMotion motion;
  GenericFit(features, &MotionEstimation::EstimateLinearSimilarityModel,
             &motion);
  return motion.linear_similarity();
}

AffineModel FitAffineModel(const RegionFlowFeatureList& features) {
  CameraMotion motion;
  GenericFit(features, &MotionEstimation::EstimateAffineModel, &motion);
  return motion.affine();
}

Homography FitHomography(const RegionFlowFeatureList& features) {
  CameraMotion motion;
  GenericFit(features, &MotionEstimation::EstimateHomography, &motion);
  return motion.homography();
}

MixtureHomography FitMixtureHomography(const RegionFlowFeatureList& features) {
  CameraMotion motion;
  GenericFit(features, &MotionEstimation::EstimateMixtureHomography, &motion);
  return motion.mixture_homography();
}

// Records inlier state across frames.
// Specifically records spatial position and average magnitude of inliers
// over time (motion prior).
// New sample points can be weighted w.r.t. their agreement of spatial inlier
// locations and motion prior.
class InlierMask {
 public:
  // Initialize mask from options for specified frame domain.
  InlierMask(const MotionEstimationOptions::IrlsMaskOptions& options,
             int feature_mask_size, int frame_width, int frame_height)
      : options_(options),
        frame_width_(frame_width),
        frame_height_(frame_height) {
    const int num_bins = feature_mask_size * feature_mask_size;
    mask_.resize(num_bins);
    update_mask_.resize(num_bins);

    const LinearSimilarityModel norm_model =
        LinearSimilarityAdapter::NormalizationTransform(frame_width_,
                                                        frame_height_);
    const Vector2_f domain = LinearSimilarityAdapter::TransformPoint(
        norm_model, Vector2_f(frame_width_, frame_height_));
    denom_x_ = 1.0f / domain.x();
    denom_y_ = 1.0f / domain.y();
    base_score_ = options_.base_score();
  }

  // Resets mask to all inliers.
  void InitMask() {
    mask_.assign(mask_.size(), 1.0f);
    translation_prior_ = 0;
  }

  // Applies update mask to mask.
  void UpdateMask() { update_mask_.swap(mask_); }

  void UpdateTranslation(Vector2_f translation) {
    const float alpha = options_.translation_blend_alpha() * translation_prior_;
    translation_ = translation_ * alpha + (1.0 - alpha) * translation;
    translation_prior_ = std::min(
        1.0f, translation_prior_ + options_.translation_prior_increase());
  }

  // Initialize update mask from current mask, by decaying each element.
  void InitUpdateMask() {
    const float decay = options_.decay();
    for (int k = 0; k < mask_.size(); ++k) {
      update_mask_[k] = mask_[k] * decay;
    }
  }

  // Returns inlier score for bin index.
  // Can be > 1, as we take the best inlier score compared to other iterations,
  // only relative values matter.
  float GetInlierScore(int idx) const { return base_score_ + mask_[idx]; }

  // Increases inlier score at bin idx.
  void RecordInlier(int idx, float feature_weight) {
    update_mask_[idx] = std::min(
        1.0f, update_mask_[idx] + feature_weight * options_.inlier_score());
  }

  // Multiplies passed motion prior with a weight within [0, 1] for each
  // feature point describing how well feature's motion agrees with previously
  // estimated translation.
  void MotionPrior(const RegionFlowFeatureList& feature_list,
                   std::vector<float>* motion_prior) {
    CHECK(motion_prior != nullptr);
    const int num_features = feature_list.feature_size();
    CHECK_EQ(num_features, motion_prior->size());

    // Return, if prior is too low.
    const float kMinTranslationPrior = 0.5f;
    if (translation_prior_ < kMinTranslationPrior) {
      motion_prior->assign(num_features, 1.0f);
      return;
    }
    const float prev_magnitude = translation_.Norm();

    CHECK_EQ(num_features, motion_prior->size());
    const float inv_prev_magnitude =
        prev_magnitude < options_.min_translation_norm()
            ? (1.0f / options_.min_translation_norm())
            : (1.0f / prev_magnitude);
    for (int k = 0; k < num_features; ++k) {
      const Vector2_f flow = FeatureFlow(feature_list.feature(k));
      const float weight =
          base_score_ + std::max<float>(0, 1.0f - (flow - translation_).Norm() *
                                                      inv_prev_magnitude);
      (*motion_prior)[k] *= weight;
    }
  }

 private:
  MotionEstimationOptions::IrlsMaskOptions options_;
  int frame_width_;
  int frame_height_;
  float denom_x_;
  float denom_y_;
  float base_score_;

  Vector2_f translation_;
  float translation_prior_ = 0;

  std::vector<float> mask_;
  std::vector<float> update_mask_;
};

// Local storage for MotionEstimation within each thread to minimize
// allocations.
class MotionEstimationThreadStorage {
 public:
  MotionEstimationThreadStorage(const MotionEstimationOptions& options,
                                const MotionEstimation* motion_estimation,
                                int max_feature_guess = 0) {
    const int coverage_grid_size = options.coverage_grid_size();
    grid_coverage_irls_mask_.resize(coverage_grid_size * coverage_grid_size);
    const int max_features = max_feature_guess > 0 ? max_feature_guess : 4000;

    // Allocate bins to 150% of expected features.
    const int features_per_bin =
        max_features * 1.5f / grid_coverage_irls_mask_.size();

    for (auto& mask : grid_coverage_irls_mask_) {
      mask.reserve(features_per_bin);
    }

    // Compute gaussian weights for grid coverage.
    const float scaled_width =
        1.0f / motion_estimation->normalized_domain_.x() * coverage_grid_size;
    const float scaled_height =
        1.0f / motion_estimation->normalized_domain_.y() * coverage_grid_size;

    const float inv_scaled_width = 1.0f / scaled_width;
    const float inv_scaled_height = 1.0f / scaled_height;

    // Compute gaussian weights for grid cells.
    RegionFlowFeatureList grid_cell_features;
    for (int y = 0; y < coverage_grid_size; ++y) {
      for (int x = 0; x < coverage_grid_size; ++x) {
        RegionFlowFeature* feature = grid_cell_features.add_feature();
        feature->set_x((x + 0.5f) * inv_scaled_width);
        feature->set_y((y + 0.5f) * inv_scaled_height);
      }
    }

    motion_estimation->GetHomographyIRLSCenterWeights(grid_cell_features,
                                                      &grid_cell_weights_);
  }

  MotionEstimationThreadStorage(const MotionEstimationThreadStorage&) = delete;
  MotionEstimationThreadStorage& operator=(
      const MotionEstimationThreadStorage&) = delete;

  std::vector<std::vector<float>>* EmptyGridCoverageIrlsMask() {
    for (auto& mask : grid_coverage_irls_mask_) {
      mask.clear();
    }
    return &grid_coverage_irls_mask_;
  }

  const std::vector<float>& GridCoverageInitializationWeights() const {
    return grid_cell_weights_;
  }

  // Creates copy of current thread storage, caller takes ownership.
  std::unique_ptr<MotionEstimationThreadStorage> Copy() const {
    std::unique_ptr<MotionEstimationThreadStorage> copy(
        new MotionEstimationThreadStorage);
    copy->grid_coverage_irls_mask_ = grid_coverage_irls_mask_;
    copy->grid_cell_weights_ = grid_cell_weights_;
    return copy;
  }

 private:
  // Empty constructor for Copy.
  MotionEstimationThreadStorage() {}

  std::vector<std::vector<float>> grid_coverage_irls_mask_;
  std::vector<float> grid_cell_weights_;
};

// Holds all the data for a clip (multiple frames) of single-frame tracks.
struct MotionEstimation::SingleTrackClipData {
  // Features to be processed. Can be set to point to external data, or
  // point to internal storage via InitializeFromInternalStorage.
  // Stores one RegionFlowFeatureList pointer per frame.
  std::vector<RegionFlowFeatureList*>* feature_lists = nullptr;

  // Camera motions to be output. Can be set to point to external data, or
  // point to internal storage via InitializeFromInternalStorage.
  // Stores one camera motion per frame.
  std::vector<CameraMotion>* camera_motions = nullptr;

  // Difference in frames that features and motions are computed for.
  int frame_diff = 1;

  // Prior weights for each frame.
  std::vector<PriorFeatureWeights> prior_weights;

  // Optional inlier mask. Used across the whole clip.
  InlierMask* inlier_mask = nullptr;

  // Weights to be passed to each stage of motion estimation.
  std::vector<std::vector<float>> irls_weight_input;

  // Indicates if weights in above vectors are uniform (to avoid testing on
  // each loop iteration).
  std::vector<bool> uniform_weight_input;

  // Indicates if non-decaying full prior should be used
  // (always bias towards initialization).
  std::vector<bool> use_full_prior;

  // Specific weights for homography.
  std::vector<std::vector<float>> homog_irls_weight_input;

  // Storage for earlier weights, in case estimated model is unstable.
  std::vector<std::vector<float>>* irls_weight_backup = nullptr;

  // If above feature_lists and camera_motions are not a view on external
  // data, storage holds underlying data.
  std::vector<RegionFlowFeatureList> feature_storage;
  std::vector<RegionFlowFeatureList*> feature_view;
  std::vector<CameraMotion> motion_storage;
  std::vector<std::vector<float>> irls_backup_storage;

  // Call after populating feature_storage and motion_storage with data, to
  // initialize feature_lists and camera_motions.
  void InitializeFromInternalStorage() {
    feature_view.reserve(feature_storage.size());

    for (auto& feature_list : feature_storage) {
      feature_view.push_back(&feature_list);
    }

    feature_lists = &feature_view;
    camera_motions = &motion_storage;
  }

  // Call after initializing feature_lists, to allocate storage for each
  // feature's irls weight. If weight_backup is set, allocates storage
  // to backup and reset irls weights.
  void AllocateIRLSWeightStorage(bool weight_backup) {
    CHECK(feature_lists != nullptr);
    const int num_frames = feature_lists->size();
    if (weight_backup) {
      irls_weight_backup = &irls_backup_storage;
    }

    if (num_frames == 0) {
      return;
    }

    irls_weight_input.resize(num_frames);
    uniform_weight_input.resize(num_frames, true);
    use_full_prior.resize(num_frames, false);
    homog_irls_weight_input.resize(num_frames);

    if (weight_backup) {
      irls_weight_backup->resize(num_frames);
    }

    for (int k = 0; k < num_frames; ++k) {
      const int num_features = (*feature_lists)[k]->feature_size();
      if (num_features != 0) {
        irls_weight_input[k].reserve(num_features);
        homog_irls_weight_input[k].reserve(num_features);
      }
    }
  }

  // Returns number of frames in this clip.
  int num_frames() const {
    DCHECK(feature_lists);
    return feature_lists->size();
  }

  // Returns irls weight input depending on the passed motion type.
  std::vector<std::vector<float>>& IrlsWeightInput(const MotionType& type) {
    switch (type) {
      case MODEL_HOMOGRAPHY:
        return homog_irls_weight_input;
      default:
        return irls_weight_input;
    }
  }

  // Checks that SingleTrackClipData is properly initialized.
  void CheckInitialization() const {
    CHECK(feature_lists != nullptr);
    CHECK(camera_motions != nullptr);
    CHECK_EQ(feature_lists->size(), camera_motions->size());
    if (feature_lists->empty()) {
      return;
    }

    CHECK_EQ(num_frames(), irls_weight_input.size());
    CHECK_EQ(num_frames(), homog_irls_weight_input.size());
    if (irls_weight_backup) {
      CHECK_EQ(num_frames(), irls_weight_backup->size());
    }

    for (int k = 0; k < num_frames(); ++k) {
      const int num_features = (*feature_lists)[k]->feature_size();
      CHECK_EQ(num_features, irls_weight_input[k].size());
      CHECK_EQ(num_features, homog_irls_weight_input[k].size());
    }
  }

  // Prepares PriorFeatureWeights structure for usage.
  void SetupPriorWeights(int irls_rounds) {
    prior_weights.resize(num_frames(), PriorFeatureWeights(irls_rounds));
    for (int k = 0; k < num_frames(); ++k) {
      prior_weights[k].use_full_prior = use_full_prior[k];
    }
  }

  // Clears the specified flag from each camera motion.
  void ClearFlagFromMotion(int flag) {
    for (auto& camera_motion : *camera_motions) {
      camera_motion.set_flags(camera_motion.flags() & ~flag);
    }
  }

  // Resets feature weights from backed up ones if type is <=
  // max_unstable_type.
  void RestoreWeightsFromBackup(CameraMotion::Type max_unstable_type) {
    if (irls_weight_backup == nullptr) {
      return;
    }

    const int num_frames = feature_lists->size();
    for (int k = 0; k < num_frames; ++k) {
      if (camera_motions->at(k).type() <= max_unstable_type) {
        SetRegionFlowFeatureIRLSWeights(irls_weight_backup->at(k),
                                        feature_lists->at(k));
      }
    }
  }
};

MotionEstimation::MotionEstimation(const MotionEstimationOptions& options,
                                   int frame_width, int frame_height)
    : frame_width_(frame_width), frame_height_(frame_height) {
  normalization_transform_ = LinearSimilarityAdapter::NormalizationTransform(
      frame_width_, frame_height_);

  inv_normalization_transform_ =
      LinearSimilarityAdapter::Invert(normalization_transform_);

  // Cap domain to express IRLS errors to 640x360 (format used
  // to calibrate thresholds on dataset).
  const int max_irls_width = frame_width_ > frame_height ? 640 : 360;
  const int max_irls_height = frame_width_ > frame_height ? 360 : 640;
  const int irls_width = std::min(max_irls_width, frame_width_);
  const int irls_height = std::min(max_irls_height, frame_height_);
  irls_transform_ = ModelInvert(
      LinearSimilarityAdapter::NormalizationTransform(irls_width, irls_height));
  if (!options.domain_limited_irls_scaling()) {
    // Fallback to inverse normalization transform, i.e. express errors
    // in image domain.
    irls_transform_ = inv_normalization_transform_;
  }

  normalized_domain_ = TransformPoint(normalization_transform_,
                                      Vector2_f(frame_width_, frame_height_));

  InitializeWithOptions(options);
}

MotionEstimation::~MotionEstimation() {}

void MotionEstimation::InitializeWithOptions(
    const MotionEstimationOptions& options) {
  // Check options, specifically if fall-back models are set to be estimated.
  if (options.homography_estimation() !=
          MotionEstimationOptions::ESTIMATION_HOMOG_NONE &&
      options.linear_similarity_estimation() ==
          MotionEstimationOptions::ESTIMATION_LS_NONE) {
    LOG(FATAL) << "Invalid MotionEstimationOptions. "
               << "Homography estimation requires similarity to be estimated";
  }

  if (options.mix_homography_estimation() !=
          MotionEstimationOptions::ESTIMATION_HOMOG_MIX_NONE &&
      options.homography_estimation() ==
          MotionEstimationOptions::ESTIMATION_HOMOG_NONE) {
    LOG(FATAL) << "Invalid MotionEstimationOptions. "
               << "Mixture homography estimation requires homography to be "
               << "estimated.";
  }

  // Check for deprecated options.
  CHECK_NE(options.estimate_similarity(), true)
      << "Option estimate_similarity is deprecated, use static function "
      << "EstimateSimilarityModelL2 instead.";
  CHECK_NE(options.linear_similarity_estimation(),
           MotionEstimationOptions::ESTIMATION_LS_L2_RANSAC)
      << "Option ESTIMATION_LS_L2_RANSAC is deprecated, use "
      << "ESTIMATION_LS_IRLS instead.";
  CHECK_NE(options.linear_similarity_estimation(),
           MotionEstimationOptions::ESTIMATION_LS_L1)
      << "Option ESTIMATION_LS_L1 is deprecated, use static function "
      << "EstimateLinearSimilarityL1 instead.";

  options_ = options;

  // (Re)-Initialize row_weights_ based on options.
  if (options.mix_homography_estimation() !=
      MotionEstimationOptions::ESTIMATION_HOMOG_MIX_NONE) {
    const float row_sigma = options.mixture_row_sigma() * frame_height_;
    const float y_scale = frame_height_ / normalized_domain_.y();

    if (row_weights_ == NULL ||
        row_weights_->NeedsInitialization(options.num_mixtures(), row_sigma,
                                          y_scale)) {
      row_weights_.reset(new MixtureRowWeights(frame_height_,
                                               0,  // no margin.
                                               row_sigma, y_scale,
                                               options.num_mixtures()));
    }
  }

  switch (options.estimation_policy()) {
    case MotionEstimationOptions::INDEPENDENT_PARALLEL:
    case MotionEstimationOptions::JOINTLY_FROM_TRACKS:
      break;

    case MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS: {
      const auto& bias_options = options.long_feature_bias_options();

      // Using 3x3 filters, max distance is 2 bin diagonals plus 1% room
      // incase maximum value is attained.
      const float max_space_diff =
          2.0f * sqrt(2.0) * bias_options.grid_size() * 1.01f;

      InitGaussLUT(bias_options.spatial_sigma(), max_space_diff,
                   &feature_bias_lut_.spatial_lut,
                   &feature_bias_lut_.spatial_scale);

      const float max_color_diff =
          std::sqrt(static_cast<double>(3.0f)) * 255.0f;  // 3 channels.
      InitGaussLUT(bias_options.color_sigma(), max_color_diff,
                   &feature_bias_lut_.color_lut,
                   &feature_bias_lut_.color_scale);

      // Gaussian at 2.5 (normalized) < 0.05
      const float max_weight = bias_options.bias_stdev() * 2.5 * 1.01f;
      InitGaussLUT(bias_options.bias_stdev(), max_weight,
                   &feature_bias_lut_.bias_weight_lut,
                   &feature_bias_lut_.bias_weight_scale);

      break;
    }

    case MotionEstimationOptions::TEMPORAL_IRLS_MASK:
      CHECK(options.irls_initialization().activated())
          << "To use dependent_initialization, irls_initialization has to "
          << "be activated. ";
      inlier_mask_.reset(new InlierMask(options.irls_mask_options(),
                                        options.feature_mask_size(),
                                        frame_width_, frame_height_));
      inlier_mask_->InitMask();
      break;
  }
}

void MotionEstimation::EstimateMotion(const RegionFlowFrame& region_flow_frame,
                                      const int* intensity_frame,       // null
                                      const int* prev_intensity_frame,  // null
                                      CameraMotion* camera_motion) const {
  CHECK(camera_motion);

  CHECK(intensity_frame == NULL)
      << "Parameter intensity_frame is deprecated, must be NULL.";
  CHECK(prev_intensity_frame == NULL)
      << "Parameter prev_intensity_frame is deprecated, must be NULL.";

  RegionFlowFeatureList feature_list;
  GetRegionFlowFeatureList(region_flow_frame, 0, &feature_list);
  std::vector<RegionFlowFeatureList*> feature_lists(1, &feature_list);
  std::vector<CameraMotion> camera_motions(1);

  EstimateMotionsParallel(false, &feature_lists, &camera_motions);
  camera_motion->CopyFrom(camera_motions[0]);
}

bool MotionEstimation::EstimateTranslationModel(
    RegionFlowFeatureList* feature_list, CameraMotion* camera_motion) {
  EstimateTranslationModelIRLS(options_.irls_rounds(), false, feature_list,
                               nullptr, camera_motion);
  return true;
}

bool MotionEstimation::EstimateLinearSimilarityModel(
    RegionFlowFeatureList* feature_list, CameraMotion* camera_motion) {
  return EstimateLinearSimilarityModelIRLS(
      options_.irls_rounds(), false, feature_list, nullptr, camera_motion);
}

bool MotionEstimation::EstimateAffineModel(RegionFlowFeatureList* feature_list,
                                           CameraMotion* camera_motion) {
  return EstimateAffineModelIRLS(options_.irls_rounds(), feature_list,
                                 camera_motion);
}

bool MotionEstimation::EstimateHomography(RegionFlowFeatureList* feature_list,
                                          CameraMotion* camera_motion) {
  return EstimateHomographyIRLS(options_.irls_rounds(), false, nullptr, nullptr,
                                feature_list, camera_motion);
}

bool MotionEstimation::EstimateMixtureHomography(
    RegionFlowFeatureList* feature_list, CameraMotion* camera_motion) {
  return EstimateMixtureHomographyIRLS(
      options_.irls_rounds(), true, options_.mixture_regularizer(),
      0,  // spectrum index.
      nullptr, nullptr, feature_list, camera_motion);
}

float MotionEstimation::GetIRLSResidualScale(const float avg_motion_magnitude,
                                             float distance_fraction) const {
  const float translation_magnitude =
      LinearSimilarityAdapter::TransformPoint(
          normalization_transform_, Vector2_f(avg_motion_magnitude, 0))
          .x();

  // Assume 1 pixel estimation error for tracked features at 480p video.
  // This serves as absolute minimum of the estimation error, so we do not
  // scale translation fractions below this threshold.
  const float minimum_error = 1.25e-3f;

  // Only normalize if residual (w.r.t. translation) is larger than
  // estimation error.
  const float translation_threshold = translation_magnitude * distance_fraction;
  // Translation is above minimum error.
  if (translation_threshold > minimum_error) {
    return minimum_error / translation_threshold;
  } else {
    return 1.0f;
  }
}

// Collects various options that alter how motion models are initialized or
// estimated. Construct from MotionEstimationOptions for default values.
struct MotionEstimation::EstimateModelOptions {
  explicit EstimateModelOptions(const MotionEstimationOptions& options)
      : mixture_regularizer(options.mixture_regularizer()),
        estimate_linear_similarity(
            options.linear_similarity_estimation() !=
            MotionEstimationOptions::ESTIMATION_LS_NONE) {}

  // Maps each motion type to a unique index, whereas different mixtures in a
  // spectrum are treated as separate types.
  int IndexFromType(const MotionType& type) const {
    if (type != MODEL_MIXTURE_HOMOGRAPHY) {
      return static_cast<int>(type);
    } else {
      return static_cast<int>(type) + mixture_spectrum_index;
    }
  }

  float mixture_regularizer = 0;
  float mixture_inlier_threshold_scale = 0;
  int mixture_spectrum_index = 0;
  bool check_model_stability = true;
  bool estimate_linear_similarity = true;
};

// Invoker for parallel execution. Thread storage is optional.
class EstimateMotionIRLSInvoker {
 public:
  // Performs estimation of the requested type for irls_rounds IRLS iterations.
  // Only performs estimation if current motion type is less or equal to
  // max_unstable_type.
  // Optionally accepts list of PriorFeatureWeights and thread_storage.
  // After model computation irls_weight member for each RegionFlowFeature
  // in the passed feature_lists is updated with the inverse fitting error.
  // Stability features for requested MotionType are computed if
  // compute_stability argument is set.
  EstimateMotionIRLSInvoker(
      const MotionEstimation::MotionType& type, int irls_rounds,
      bool compute_stability, const CameraMotion::Type& max_unstable_type,
      const MotionEstimation::EstimateModelOptions& model_options,
      const MotionEstimation* motion_estimation,
      const std::vector<MotionEstimation::PriorFeatureWeights>*
          prior_weights,                                    // optional.
      const MotionEstimationThreadStorage* thread_storage,  // optional.
      std::vector<RegionFlowFeatureList*>* feature_lists,
      std::vector<CameraMotion>* camera_motions)
      : motion_type_(type),
        irls_rounds_(irls_rounds),
        compute_stability_(compute_stability),
        max_unstable_type_(max_unstable_type),
        model_options_(model_options),
        motion_estimation_(motion_estimation),
        prior_weights_(prior_weights),
        feature_lists_(feature_lists),
        camera_motions_(camera_motions) {
    if (thread_storage != nullptr) {
      std::unique_ptr<MotionEstimationThreadStorage> tmp_storage(
          thread_storage->Copy());
      thread_storage_ = std::move(tmp_storage);
    }
  }

  EstimateMotionIRLSInvoker(const EstimateMotionIRLSInvoker& invoker)
      : motion_type_(invoker.motion_type_),
        irls_rounds_(invoker.irls_rounds_),
        compute_stability_(invoker.compute_stability_),
        max_unstable_type_(invoker.max_unstable_type_),
        model_options_(invoker.model_options_),
        motion_estimation_(invoker.motion_estimation_),
        prior_weights_(invoker.prior_weights_),
        feature_lists_(invoker.feature_lists_),
        camera_motions_(invoker.camera_motions_) {
    if (invoker.thread_storage_ != nullptr) {
      std::unique_ptr<MotionEstimationThreadStorage> tmp_storage(
          invoker.thread_storage_->Copy());
      thread_storage_ = std::move(tmp_storage);
    }
  }

  void operator()(const BlockedRange& range) const {
    for (int frame = range.begin(); frame != range.end(); ++frame) {
      EstimateMotion(frame, (*feature_lists_)[frame],
                     &(*camera_motions_)[frame]);
    }
  }

 private:
  inline void EstimateMotion(int frame, RegionFlowFeatureList* feature_list,
                             CameraMotion* camera_motion) const {
    if (camera_motion->type() > max_unstable_type_) {
      // Don't estimate anything, immediate return.
      return;
    }

    if (camera_motion->flags() & CameraMotion::FLAG_SINGULAR_ESTIMATION) {
      // If motion became singular in earlier iteration, skip.
      return;
    }

    const MotionEstimation::PriorFeatureWeights* prior_weight =
        prior_weights_ && (*prior_weights_)[frame].HasPrior()
            ? &(*prior_weights_)[frame]
            : nullptr;

    switch (motion_type_) {
      case MotionEstimation::MODEL_AVERAGE_MAGNITUDE:
        motion_estimation_->EstimateAverageMotionMagnitude(*feature_list,
                                                           camera_motion);
        break;

      case MotionEstimation::MODEL_TRANSLATION:
        motion_estimation_->EstimateTranslationModelIRLS(
            irls_rounds_, compute_stability_, feature_list, prior_weight,
            camera_motion);
        break;

      case MotionEstimation::MODEL_LINEAR_SIMILARITY:
        motion_estimation_->EstimateLinearSimilarityModelIRLS(
            irls_rounds_, compute_stability_, feature_list, prior_weight,
            camera_motion);
        break;

      case MotionEstimation::MODEL_AFFINE:
        motion_estimation_->EstimateAffineModelIRLS(irls_rounds_, feature_list,
                                                    camera_motion);
        break;

      case MotionEstimation::MODEL_HOMOGRAPHY:
        motion_estimation_->EstimateHomographyIRLS(
            irls_rounds_, compute_stability_, prior_weight,
            thread_storage_.get(), feature_list, camera_motion);
        break;

      case MotionEstimation::MODEL_MIXTURE_HOMOGRAPHY:
        // If one estimation fails, clear the whole spectrum.
        if (!motion_estimation_->EstimateMixtureHomographyIRLS(
                irls_rounds_, compute_stability_,
                model_options_.mixture_regularizer,
                model_options_.mixture_spectrum_index, prior_weight,
                thread_storage_.get(), feature_list, camera_motion)) {
          camera_motion->clear_mixture_homography_spectrum();
        }
        break;

      case MotionEstimation::MODEL_NUM_VALUES:
        LOG(FATAL) << "Function should not be called with this value";
        break;
    }
  }

 private:
  const MotionEstimation::MotionType motion_type_;
  int irls_rounds_;
  bool compute_stability_;
  CameraMotion::Type max_unstable_type_;
  const MotionEstimation::EstimateModelOptions& model_options_;
  const MotionEstimation* motion_estimation_;
  const std::vector<MotionEstimation::PriorFeatureWeights>* prior_weights_;
  std::vector<RegionFlowFeatureList*>* feature_lists_;
  std::vector<CameraMotion>* camera_motions_;

  std::unique_ptr<MotionEstimationThreadStorage> thread_storage_;
};

void MotionEstimation::EstimateMotionsParallelImpl(
    bool irls_weights_preinitialized,
    std::vector<RegionFlowFeatureList*>* feature_lists,
    std::vector<CameraMotion>* camera_motions) const {
  MEASURE_TIME << "Estimate motions: " << feature_lists->size();

  CHECK(feature_lists != nullptr);
  CHECK(camera_motions != nullptr);

  const int num_frames = feature_lists->size();
  CHECK_EQ(num_frames, camera_motions->size());

  // Initialize camera_motions.
  for (int f = 0; f < num_frames; ++f) {
    CameraMotion& camera_motion = (*camera_motions)[f];
    const RegionFlowFeatureList& feature_list = *(*feature_lists)[f];

    // Resets every model to INVALID.
    ResetMotionModels(options_, &camera_motion);
    InitCameraMotionFromFeatureList(feature_list, &camera_motion);

    // Assume motions to be VALID in case they contain features.
    // INVALID (= empty frames) wont be considered during motion estimation.
    if (feature_list.feature_size() != 0) {
      camera_motion.set_type(CameraMotion::VALID);
    }

    // Flag duplicated frames.
    if (feature_list.is_duplicated()) {
      camera_motion.set_flags(camera_motion.flags() |
                              CameraMotion::FLAG_DUPLICATED);
    }
  }

  // Backup original IRLS weights if original weights are requested to be
  // output.
  std::vector<std::vector<float>> original_irls_weights(num_frames);
  if (!options_.output_refined_irls_weights()) {
    for (int f = 0; f < num_frames; ++f) {
      const RegionFlowFeatureList& feature_list = *(*feature_lists)[f];
      GetRegionFlowFeatureIRLSWeights(feature_list, &original_irls_weights[f]);
    }
  }

  const bool use_joint_tracks = options_.estimation_policy() ==
                                MotionEstimationOptions::JOINTLY_FROM_TRACKS;

  // Joint frame estimation.
  const int num_motion_models =
      use_joint_tracks ? options_.joint_track_estimation().num_motion_models()
                       : 1;
  CHECK_GT(num_motion_models, 0);

  // Several single track clip datas, we seek to process.
  std::vector<SingleTrackClipData> clip_datas(num_motion_models);
  SingleTrackClipData* main_clip_data = &clip_datas[0];

  // First clip data is always view on external data.
  main_clip_data->feature_lists = feature_lists;
  main_clip_data->camera_motions = camera_motions;
  main_clip_data->inlier_mask = inlier_mask_.get();
  main_clip_data->frame_diff = 1;

  main_clip_data->AllocateIRLSWeightStorage(true);

  LongFeatureInfo long_feature_info;

  if (irls_weights_preinitialized &&
      options_.filter_initialized_irls_weights()) {
    MinFilterIrlsWeightByTrack(main_clip_data);
  }

  // Determine importance for each track length.
  std::vector<float> track_length_importance;
  if (options_.long_feature_initialization().activated()) {
    for (const auto feature_list_ptr : *feature_lists) {
      if (feature_list_ptr->long_tracks()) {
        long_feature_info.AddFeatures(*feature_list_ptr);
      }
    }

    const float percentile =
        options_.long_feature_initialization().min_length_percentile();
    const float min_length = long_feature_info.GlobalTrackLength(percentile);

    track_length_importance.resize(num_frames + 1, 1.0f);
    // Gaussian weighting.
    const float denom = -0.5f / (2.0f * 2.0f);  // 2 frame stdev.
    for (int k = 0; k <= num_frames; ++k) {
      float weight = 1.0f;
      if (k < min_length) {
        weight = std::exp((k - min_length) * (k - min_length) * denom);
      }
      track_length_importance[k] = weight;
    }
  }

  int max_features = 0;
  for (int f = 0; f < num_frames; ++f) {
    // Initialize IRLS input.
    const RegionFlowFeatureList& feature_list =
        *(*main_clip_data->feature_lists)[f];

    max_features = std::max(feature_list.feature_size(), max_features);

    std::vector<float>& irls_weight_input =
        main_clip_data->irls_weight_input[f];

    if (irls_weights_preinitialized) {
      GetRegionFlowFeatureIRLSWeights(feature_list, &irls_weight_input);
    } else {
      // Set to one.
      irls_weight_input.resize(feature_list.feature_size(), 1.0f);
    }

    // Note: To create visualizations of the prior stored in irls_weight_input,
    // add a call to
    // SetRegionFlowFeatureIRLSWeights(irls_weight_input, &feature_list);
    // anywhere below and adopt irls_rounds to zero in options.
    // This will effectively output identity motions with irls weights set
    // to the priors here.
    bool uniform_weights = !irls_weights_preinitialized;
    bool use_full_prior = false;

    if (options_.long_feature_initialization().activated()) {
      if (!feature_list.long_tracks()) {
        LOG(ERROR) << "Requesting long feature initialization but "
                   << "input is not computed with long features.";
      } else {
        LongFeatureInitialization(feature_list, long_feature_info,
                                  track_length_importance, &irls_weight_input);
        uniform_weights = false;
        use_full_prior = true;
      }
    }

    if (options_.feature_density_normalization()) {
      FeatureDensityNormalization(feature_list, &irls_weight_input);
      uniform_weights = false;
      use_full_prior = true;
    }

    GetHomographyIRLSCenterWeights(feature_list,
                                   &main_clip_data->homog_irls_weight_input[f]);

    if (!uniform_weights) {
      // Multiply homography weights by non-uniform irls input weights.
      std::vector<float>& homg_weights =
          main_clip_data->homog_irls_weight_input[f];

      const int num_features = feature_list.feature_size();
      for (int k = 0; k < num_features; ++k) {
        homg_weights[k] *= irls_weight_input[k];
      }
    }

    main_clip_data->uniform_weight_input[f] = uniform_weights;
    main_clip_data->use_full_prior[f] = use_full_prior;
  }

  if (options_.estimation_policy() ==
      MotionEstimationOptions::JOINTLY_FROM_TRACKS) {
    for (int k = 1; k < num_motion_models; ++k) {
      SingleTrackClipData* curr_clip_data = &clip_datas[k];
      // Camera motions are simply a copy of the initialized main data.
      curr_clip_data->motion_storage = *camera_motions;
      curr_clip_data->feature_storage.resize(num_frames);
      curr_clip_data->InitializeFromInternalStorage();

      // No backup storage for non-main data (feature's are not output).
      curr_clip_data->AllocateIRLSWeightStorage(false);
      const int stride = options_.joint_track_estimation().motion_stride();
      curr_clip_data->frame_diff = stride * k;

      // Populate feature and motion storage.
      for (int f = 0; f < num_frames; ++f) {
        const int prev_frame = f - stride * k;
        if (prev_frame < 0) {
          CopyToEmptyFeatureList((*feature_lists)[f],
                                 &curr_clip_data->feature_storage[f]);
        } else {
          // Determine features present in both frames along the long feature
          // tracks.
          std::vector<int> source_idx;
          IntersectRegionFlowFeatureList(*(*feature_lists)[prev_frame],
                                         &FeatureLocation, (*feature_lists)[f],
                                         &curr_clip_data->feature_storage[f],
                                         &source_idx);

          curr_clip_data->irls_weight_input[f].reserve(source_idx.size());
          curr_clip_data->homog_irls_weight_input[f].reserve(source_idx.size());

          // Copy weights.
          for (int idx : source_idx) {
            curr_clip_data->irls_weight_input[f].push_back(
                main_clip_data->irls_weight_input[f][idx]);
            curr_clip_data->homog_irls_weight_input[f].push_back(
                main_clip_data->homog_irls_weight_input[f][idx]);
          }
        }
      }
    }
  }

  for (auto& clip_data : clip_datas) {
    clip_data.CheckInitialization();
  }

  for (auto& clip_data : clip_datas) {
    // Estimate AverageMotion magnitudes.
    ParallelFor(0, num_frames, 1,
                EstimateMotionIRLSInvoker(
                    MODEL_AVERAGE_MAGNITUDE,
                    1,     // Does not use irls.
                    true,  // Compute stability.
                    CameraMotion::VALID, DefaultModelOptions(), this,
                    nullptr,  // No prior weights.
                    nullptr,  // No thread storage.
                    clip_data.feature_lists, clip_data.camera_motions));
  }

  // Order of estimation for motion models:
  // Translation -> Linear Similarity -> Affine -> Homography -> Mixture
  // Homography.

  // Estimate translations, regardless of stability of similarity.
  EstimateMotionModels(MODEL_TRANSLATION,
                       CameraMotion::UNSTABLE,  // Any but invalid.
                       DefaultModelOptions(),
                       nullptr,  // No thread storage.
                       &clip_datas);

  // Estimate linear similarity, but only if translation was deemed stable.
  EstimateMotionModels(MODEL_LINEAR_SIMILARITY, CameraMotion::VALID,
                       DefaultModelOptions(),
                       nullptr,  // No thread storage.
                       &clip_datas);

  if (options_.project_valid_motions_down()) {
    ProjectMotionsDown(MODEL_LINEAR_SIMILARITY, camera_motions);
  }

  // Estimate affine, but only if similarity was deemed stable.
  EstimateMotionModels(MODEL_AFFINE, CameraMotion::VALID, DefaultModelOptions(),
                       nullptr,  // No thread storage.
                       &clip_datas);

  // Thread storage below is only used for homography or mixtures.
  MotionEstimationThreadStorage thread_storage(options_, this, max_features);

  // Estimate homographies, only if similarity was deemed stable.
  EstimateMotionModels(MODEL_HOMOGRAPHY, CameraMotion::VALID,
                       DefaultModelOptions(), &thread_storage, &clip_datas);

  if (options_.project_valid_motions_down()) {
    // If homography is unstable, then whatever was deemed stable got
    // embedded here and is now down projected again.
    ProjectMotionsDown(MODEL_HOMOGRAPHY, camera_motions);
  }

  // Estimate mixtures. We attempt estimation as long as at least
  // translation was estimated stable.
  // Estimate mixtures across a spectrum a different regularizers, from the
  // weakest to the most regularized one.
  const int num_mixture_levels = options_.mixture_regularizer_levels();
  CHECK_LE(num_mixture_levels, 10) << "Only up to 10 mixtures are supported.";

  // Initialize to weakest regularizer.
  float regularizer = options_.mixture_regularizer();

  // Initialize with maximum of weakest regularized mixture.
  float inlier_threshold_scale =
      std::pow(static_cast<double>(options_.mixture_regularizer_base()),
               static_cast<double>(options_.mixture_regularizer_levels() - 1));

  bool base_mixture_estimated = false;
  for (int m = 0; m < num_mixture_levels; ++m) {
    EstimateModelOptions options = DefaultModelOptions();
    options.mixture_regularizer = regularizer;
    options.mixture_inlier_threshold_scale = inlier_threshold_scale;
    options.mixture_spectrum_index = m;
    // Only check stability for weakest regularized mixture.
    options.check_model_stability = m == 0;
    // Estimate weakest mixture even if similarity was deemed unstable, higher
    // mixtures only if weakest mixture was deemed stable.
    const bool estimate_result = EstimateMotionModels(
        MODEL_MIXTURE_HOMOGRAPHY,
        m == 0 ? CameraMotion::UNSTABLE : CameraMotion::VALID, options,
        &thread_storage, &clip_datas);

    if (m == 0) {
      base_mixture_estimated = estimate_result;
    }

    regularizer *= options_.mixture_regularizer_base();
    inlier_threshold_scale /= options_.mixture_regularizer_base();

    // Preserve IRLS weights from the very first mixture (all stability is
    // computed w.r.t. it).
    if (base_mixture_estimated && m > 0) {
      for (auto& clip_data : clip_datas) {
        clip_data.RestoreWeightsFromBackup(CameraMotion::VALID);
      }
    }
  }

  // Check that mixture spectrum has sufficient entries.
  for (const CameraMotion& motion : *camera_motions) {
    if (motion.mixture_homography_spectrum_size() > 0) {
      CHECK_EQ(motion.mixture_homography_spectrum_size(),
               options_.mixture_regularizer_levels());
    }
  }

  IRLSWeightFilter(feature_lists);

  if (!options_.output_refined_irls_weights()) {
    for (int f = 0; f < num_frames; ++f) {
      RegionFlowFeatureList& feature_list = *(*feature_lists)[f];
      SetRegionFlowFeatureIRLSWeights(original_irls_weights[f], &feature_list);
    }
  }

  // Lift model type from INVALID for empty frames to VALID if requested.
  if (options_.label_empty_frames_as_valid()) {
    for (int f = 0; f < num_frames; ++f) {
      CameraMotion& camera_motion = (*camera_motions)[f];
      if ((*feature_lists)[f]->feature_size() == 0) {
        camera_motion.set_type(CameraMotion::VALID);
      }
    }
  }
}

MotionEstimation::EstimateModelOptions MotionEstimation::DefaultModelOptions()
    const {
  return EstimateModelOptions(options_);
}

// In the following member refers to member in SingleTrackClipData.
// For each estimation invocation, irls weights of features are set from
// member irls_weight_input.
// Motion models are estimated from member feature_list and stored in
// member camera_motions.
bool MotionEstimation::EstimateMotionModels(
    const MotionType& type, const CameraMotion::Type& max_unstable_type,
    const EstimateModelOptions& model_options,
    const MotionEstimationThreadStorage* thread_storage,
    std::vector<SingleTrackClipData>* clip_datas) const {
  CHECK(clip_datas != nullptr);

  const int num_datas = clip_datas->size();
  if (num_datas == 0) {
    return false;
  }

  for (const auto& clip_data : *clip_datas) {
    clip_data.CheckInitialization();
  }

  // Perform estimation across all frames for several total_rounds with each
  // round having irls_per_round iterations.
  int irls_per_round = 1;
  int total_rounds = 1;

  PolicyToIRLSRounds(IRLSRoundsFromSettings(type), &total_rounds,
                     &irls_per_round);

  const int total_irls_rounds = irls_per_round * total_rounds;

  // Skip if nothing to estimate.
  if (total_irls_rounds == 0) {
    return false;
  }

  // Setup each clip data for this estimation round.
  for (auto& clip_data : *clip_datas) {
    clip_data.SetupPriorWeights(irls_per_round);
    // Clear flag from earlier model estimation.
    clip_data.ClearFlagFromMotion(CameraMotion::FLAG_SINGULAR_ESTIMATION);
  }

  if (options_.estimation_policy() !=
      MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS) {
    // Irls initialization for each list.
    for (auto& clip_data : *clip_datas) {
      IrlsInitialization(type, max_unstable_type,
                         -1,  // all frames.
                         model_options, &clip_data);
    }

    // Parallel estimation across frames.
    for (int r = 0; r < total_rounds; ++r) {
      // Setup, default decaying irls alphas.
      std::vector<float> irls_alphas(irls_per_round);
      for (int k = 0; k < irls_per_round; ++k) {
        irls_alphas[k] =
            IRLSPriorWeight(r * irls_per_round + k, total_irls_rounds);
      }

      for (auto& clip_data : *clip_datas) {
        // Setup prior's alphas.
        for (auto& prior_weight : clip_data.prior_weights) {
          if (prior_weight.use_full_prior) {
            prior_weight.alphas.assign(irls_per_round, 1.0f);
          } else {
            prior_weight.alphas = irls_alphas;
          }

          // Last iteration, irls_alpha is always zero to return actual error.
          if (r + 1 == total_rounds) {
            prior_weight.alphas.back() = 0.0;
          }
        }

        const bool last_round = r + 1 == total_rounds;
        ParallelFor(0, clip_data.num_frames(), 1,
                    EstimateMotionIRLSInvoker(
                        type, irls_per_round,
                        last_round,  // Compute stability on last round.
                        max_unstable_type, model_options, this,
                        &clip_data.prior_weights, thread_storage,
                        clip_data.feature_lists, clip_data.camera_motions));
      }

      if (options_.estimation_policy() ==
          MotionEstimationOptions::JOINTLY_FROM_TRACKS) {
        EnforceTrackConsistency(clip_datas);
      }
    }

    if (model_options.check_model_stability) {
      for (auto& clip_data : *clip_datas) {
        CheckModelStability(type, max_unstable_type,
                            clip_data.irls_weight_backup,
                            clip_data.feature_lists, clip_data.camera_motions);
      }
    }
  } else {
    // Estimation policy == TEMPORAL_LONG_FEATURE_BIAS.
    for (auto& clip_data : *clip_datas) {
      EstimateMotionIRLSInvoker motion_invoker(
          type, irls_per_round,
          true,  // Compute stability on last round.
          max_unstable_type, model_options, this, &clip_data.prior_weights,
          thread_storage, clip_data.feature_lists, clip_data.camera_motions);

      for (int round = 0; round < total_rounds; ++round) {
        // Traverse frames in order.
        for (int k = 0; k < clip_data.num_frames(); ++k) {
          if (clip_data.feature_lists->at(k)->feature_size() > 0) {
            CHECK(clip_data.feature_lists->at(k)->long_tracks())
                << "Estimation policy TEMPORAL_LONG_FEATURE_BIAS requires "
                << "tracking with long tracks.";
          }

          // First round -> initialize weights.
          if (round == 0) {
            IrlsInitialization(type, max_unstable_type, k, model_options,
                               &clip_data);

            BiasLongFeatures(clip_data.feature_lists->at(k), type,
                             model_options, &clip_data.prior_weights[k]);
          }

          if (clip_data.camera_motions->at(k).type() <= max_unstable_type) {
            CHECK(clip_data.prior_weights[k].use_full_prior);
            clip_data.prior_weights[k].alphas.assign(irls_per_round, 1.0f);
            clip_data.prior_weights[k].alphas.back() = 0.0;
          }

          // Compute per-frame motion.
          motion_invoker(BlockedRange(k, k + 1, 1));

          if (model_options.check_model_stability) {
            CheckSingleModelStability(type, max_unstable_type,
                                      clip_data.irls_weight_backup
                                          ? &clip_data.irls_weight_backup->at(k)
                                          : nullptr,
                                      clip_data.feature_lists->at(k),
                                      &clip_data.camera_motions->at(k));
          }

          if (clip_data.camera_motions->at(k).type() == CameraMotion::VALID) {
            const bool remove_terminated_tracks =
                total_rounds == 1 || (round == 0 && k == 0);
            UpdateLongFeatureBias(type, model_options, remove_terminated_tracks,
                                  round > 0,  // Update irls observation.
                                  clip_data.feature_lists->at(k));
          }
        }

        // Update feature weights and priors for the next round.
        for (int k = 0; k < clip_data.num_frames(); ++k) {
          auto& feats = *clip_data.feature_lists->at(k);
          auto& priors = clip_data.prior_weights[k].priors;
          const int type_idx = model_options.IndexFromType(type);
          const auto& bias_map = long_feature_bias_maps_[type_idx];

          for (int l = 0; l < feats.feature_size(); ++l) {
            auto iter = bias_map.find(feats.feature(l).track_id());
            if (iter != bias_map.end()) {
              const float bias = iter->second.bias;
              float irls = 1.0f / (bias + kIrlsEps);
              if (irls < 1.0f) {
                irls *= irls;  // Downweight outliers even more.
              }
              feats.mutable_feature(l)->set_irls_weight(irls);
              priors[l] = irls;
            }
          }
        }
      }
    }
  }
  return true;
}

namespace {

class DoesFeatureAgreeWithSimilarity {
 public:
  DoesFeatureAgreeWithSimilarity(const LinearSimilarityModel& similarity,
                                 float inlier_threshold)
      : similarity_(similarity),
        sq_inlier_threshold_(inlier_threshold * inlier_threshold) {}
  bool operator()(const RegionFlowFeature& feature) const {
    Vector2_f lin_pt = LinearSimilarityAdapter::TransformPoint(
        similarity_, FeatureLocation(feature));
    return (lin_pt - FeatureMatchLocation(feature)).Norm2() <
           sq_inlier_threshold_;
  }

 private:
  LinearSimilarityModel similarity_;
  float sq_inlier_threshold_;
};

}  // namespace.

class IrlsInitializationInvoker {
 public:
  IrlsInitializationInvoker(
      const MotionEstimation::MotionType& type, int max_unstable_type,
      const MotionEstimation::EstimateModelOptions& model_options,
      const MotionEstimation* motion_estimation,
      MotionEstimation::SingleTrackClipData* clip_data)
      : type_(type),
        max_unstable_type_(max_unstable_type),
        model_options_(model_options),
        motion_estimation_(motion_estimation),
        clip_data_(clip_data) {}

  void operator()(const BlockedRange& range) const {
    for (int frame = range.begin(); frame != range.end(); ++frame) {
      const CameraMotion& camera_motion = (*clip_data_->camera_motions)[frame];
      RegionFlowFeatureList& feature_list =
          *(*clip_data_->feature_lists)[frame];

      // Don't process motions that are already deemed unstable.
      // Keep last resulting weights around.
      if (camera_motion.type() > max_unstable_type_) {
        continue;
      }

      // Backup irls weights, reset of weights occurs in CheckModelStability
      // if model was deemed unstable.
      if (clip_data_->irls_weight_backup) {
        GetRegionFlowFeatureIRLSWeights(
            feature_list, &(*clip_data_->irls_weight_backup)[frame]);
      }

      // Should translation or similarity irls weights be initialized?
      // In this case, prior weights will be enforced during estimation.
      const bool irls_initialization =
          motion_estimation_->options_.irls_initialization().activated();

      // Should (mixture) homography irls weights be initialized?
      // In this case, prior weights are not necessarily enforced, to allow
      // estimation to deviate from filtered result.
      const bool use_only_lin_sim_inliers_for_homography =
          motion_estimation_->options_
              .use_only_lin_sim_inliers_for_homography();

      // Only seed priors if at least one round of translation estimation was
      // performed, i.e. feature weights contain results of that estimation.
      bool seed_priors_from_bias =
          motion_estimation_->options_.estimation_policy() ==
              MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS &&
          motion_estimation_->options_.long_feature_bias_options()
              .seed_priors_from_bias() &&
          type_ >= MotionEstimation::MODEL_LINEAR_SIMILARITY;

      if (seed_priors_from_bias &&
          type_ == MotionEstimation::MODEL_LINEAR_SIMILARITY) {
        // Check that translation variance is not too large, otherwise
        // biasing from prior distorts results (think of zooming, rotation,
        // etc.).
        std::vector<float> variances;
        for (const auto& camera_motion : *clip_data_->camera_motions) {
          variances.push_back(camera_motion.translation_variance());
        }
        auto percentile = variances.begin() + variances.size() * 0.8f;
        std::nth_element(variances.begin(), percentile, variances.end());
        const float variance =
            *percentile / hypot(motion_estimation_->frame_width_,
                                motion_estimation_->frame_height_);
        constexpr float kMaxTranslationVariance = 5e-3;
        seed_priors_from_bias = variance < kMaxTranslationVariance;
      }

      if (seed_priors_from_bias &&
          type_ == MotionEstimation::MODEL_HOMOGRAPHY) {
        // Check strict inlier ratio; if there are sufficient inliers
        // don't bias homography to better fit; otherwise bias for more
        // rigidity.
        std::vector<float> inlier_ratio;
        for (const auto& camera_motion : *clip_data_->camera_motions) {
          inlier_ratio.push_back(
              camera_motion.similarity_strict_inlier_ratio());
        }
        auto percentile = inlier_ratio.begin() + inlier_ratio.size() * 0.5f;
        std::nth_element(inlier_ratio.begin(), percentile, inlier_ratio.end());
        // TODO: Revisit decision boundary after limiting
        // motion estimation to selection.
        constexpr float kMaxRatio = 0.7;
        seed_priors_from_bias = *percentile < kMaxRatio;
      }

      // Seed priors from previous estimation if requested.
      if (seed_priors_from_bias) {
        GetRegionFlowFeatureIRLSWeights(
            feature_list, &clip_data_->prior_weights[frame].priors);
      }

      // Initialize irls weights to their default values.
      SetRegionFlowFeatureIRLSWeights(clip_data_->IrlsWeightInput(type_)[frame],
                                      &feature_list);

      // Update weights based on type and options.
      // Initialization step.
      switch (type_) {
        case MotionEstimation::MODEL_TRANSLATION:
          if (irls_initialization) {
            TranslationModel best_model;
            if (motion_estimation_->GetTranslationIrlsInitialization(
                    &feature_list, model_options_,
                    camera_motion.average_magnitude(), clip_data_->inlier_mask,
                    &best_model)) {
              // Successful initialization, update inlier mask and signal to
              // use full prior.
              clip_data_->prior_weights[frame].use_full_prior = true;
              if (clip_data_->inlier_mask) {
                clip_data_->inlier_mask->UpdateTranslation(
                    Vector2_f(best_model.dx(), best_model.dy()));
                clip_data_->inlier_mask->UpdateMask();
              }
            } else {
              // Initialization failed, reset to original weights.
              SetRegionFlowFeatureIRLSWeights(
                  clip_data_->IrlsWeightInput(type_)[frame], &feature_list);

              if (clip_data_->inlier_mask) {
                // Unstable translation: Pretty bad motion here, should reset
                // prior.
                clip_data_->inlier_mask->InitMask();
              }
            }
          }
          break;

        case MotionEstimation::MODEL_LINEAR_SIMILARITY:
          if (irls_initialization) {
            LinearSimilarityModel best_model;
            if (motion_estimation_->GetSimilarityIrlsInitialization(
                    &feature_list, model_options_,
                    camera_motion.average_magnitude(), clip_data_->inlier_mask,
                    &best_model)) {
              // Successful initialization, update inlier mask and signal to
              // use full prior.
              clip_data_->prior_weights[frame].use_full_prior = true;

              if (clip_data_->inlier_mask) {
                clip_data_->inlier_mask->UpdateMask();
              }
            } else {
              // Initialization failed, reset to original weights.
              SetRegionFlowFeatureIRLSWeights(
                  clip_data_->IrlsWeightInput(type_)[frame], &feature_list);
            }
          }
          break;

        default:
          break;
      }

      // Filtering step.
      switch (type_) {
        case MotionEstimation::MODEL_HOMOGRAPHY:
          if (use_only_lin_sim_inliers_for_homography &&
              camera_motion.type() <= CameraMotion::UNSTABLE_SIM) {
            // Threshold is set w.r.t. normalized frame diameter.
            LinearSimilarityModel normalized_similarity =
                ModelCompose3(motion_estimation_->normalization_transform_,
                              camera_motion.linear_similarity(),
                              motion_estimation_->inv_normalization_transform_);
            FilterRegionFlowFeatureList(
                DoesFeatureAgreeWithSimilarity(
                    normalized_similarity,
                    motion_estimation_->options_.lin_sim_inlier_threshold()),
                kOutlierIRLSWeight, &feature_list);
          }
          break;

        case MotionEstimation::MODEL_MIXTURE_HOMOGRAPHY:
          if (use_only_lin_sim_inliers_for_homography &&
              camera_motion.type() <= CameraMotion::UNSTABLE_SIM) {
            // Threshold is set w.r.t. normalized frame diameter.
            LinearSimilarityModel normalized_similarity =
                ModelCompose3(motion_estimation_->normalization_transform_,
                              camera_motion.linear_similarity(),
                              motion_estimation_->inv_normalization_transform_);

            // Linear similarity is a rigid motion model, only reject severe
            // outliers (4 times motion magnitude, 1.5 times rejection
            // threshold).
            const float irls_residual_scale =
                motion_estimation_->GetIRLSResidualScale(
                    camera_motion.average_magnitude(),
                    motion_estimation_->options_.irls_mixture_fraction_scale() *
                        motion_estimation_->options_
                            .irls_motion_magnitude_fraction());

            const float inlier_threshold =
                model_options_.mixture_inlier_threshold_scale *
                motion_estimation_->options_.lin_sim_inlier_threshold() /
                irls_residual_scale;

            FilterRegionFlowFeatureList(
                DoesFeatureAgreeWithSimilarity(normalized_similarity,
                                               inlier_threshold),
                kOutlierIRLSWeight, &feature_list);
          }
          break;

        default:
          break;
      }

      const bool use_prior_weights =
          !clip_data_->uniform_weight_input[frame] || irls_initialization;

      // Initialize priors from irls weights.
      if (use_prior_weights) {
        CHECK_LT(frame, clip_data_->prior_weights.size());

        if (clip_data_->prior_weights[frame].priors.empty()) {
          clip_data_->prior_weights[frame].priors.resize(
              feature_list.feature_size(), 1.0f);
        }

        if (seed_priors_from_bias) {
          std::vector<float> multiply;
          GetRegionFlowFeatureIRLSWeights(feature_list, &multiply);
          for (int l = 0; l < multiply.size(); ++l) {
            clip_data_->prior_weights[frame].priors[l] *= multiply[l];
          }
        } else {
          GetRegionFlowFeatureIRLSWeights(
              feature_list, &clip_data_->prior_weights[frame].priors);
        }
      }
    }
  }

 private:
  MotionEstimation::MotionType type_;
  int max_unstable_type_;
  const MotionEstimation::EstimateModelOptions& model_options_;
  const MotionEstimation* motion_estimation_;
  MotionEstimation::SingleTrackClipData* clip_data_;
};

void MotionEstimation::LongFeatureInitialization(
    const RegionFlowFeatureList& feature_list,
    const LongFeatureInfo& feature_info,
    const std::vector<float>& track_length_importance,
    std::vector<float>* irls_weights) const {
  CHECK(irls_weights);
  const int num_features = feature_list.feature_size();
  if (num_features == 0) {
    return;
  }

  CHECK_EQ(num_features, irls_weights->size());

  // Determine actual scale to be applied to each feature.
  std::vector<float> feature_scales(num_features);
  constexpr float kTrackLengthImportance = 0.5f;
  int num_upweighted = 0;  // Count number of upweighted features.
  for (int k = 0; k < num_features; ++k) {
    const int track_len = feature_info.TrackLength(feature_list.feature(k));
    const float track_len_scale = track_length_importance[track_len];
    if (track_len_scale >= kTrackLengthImportance) {
      ++num_upweighted;
    }

    feature_scales[k] = track_len_scale;
  }

  // Use full upweighting above kMinFraction of upweighted features.
  constexpr float kMinFraction = 0.1f;
  const float upweight_multiplier =
      options_.long_feature_initialization().upweight_multiplier() *
      std::min(1.0f, num_upweighted / (num_features * kMinFraction));

  for (int k = 0; k < num_features; ++k) {
    // Never downweight.
    (*irls_weights)[k] *=
        std::max(1.0f, feature_scales[k] * upweight_multiplier);
  }
}

void MotionEstimation::FeatureDensityNormalization(
    const RegionFlowFeatureList& feature_list,
    std::vector<float>* irls_weights) const {
  CHECK(irls_weights);
  const int num_features = feature_list.feature_size();
  CHECK_EQ(num_features, irls_weights->size());

  // Compute mask index for each feature.
  std::vector<int> bin_indices;
  bin_indices.reserve(num_features);

  const int mask_size = options_.feature_mask_size();
  const int max_bins = mask_size * mask_size;

  std::vector<float> bin_normalizer(max_bins, 0.0f);

  const Vector2_f domain = NormalizedDomain();
  const float scale_x = (mask_size - 1) / domain.x();
  const float scale_y = (mask_size - 1) / domain.y();

  // Interpolate location into adjacent bins.
  for (const auto& feature : feature_list.feature()) {
    const float grid_y = feature.y() * scale_y;
    const float grid_x = feature.x() * scale_x;

    const int int_grid_x = grid_x;
    const int int_grid_y = grid_y;

    const float dx = grid_x - int_grid_x;
    const float dy = grid_y - int_grid_y;
    const float dxdy = dx * dy;
    const float dx_plus_dy = dx + dy;

    const int inc_x = dx != 0;
    const int inc_y = dy != 0;

    int bin_idx = int_grid_y * mask_size + int_grid_x;
    // (1 - dx)(1 - dy) = 1 - (dx + dy) + dx*dy
    bin_normalizer[bin_idx] += 1 - dx_plus_dy + dxdy;
    // dx * (1 - dy) = dx - dxdy
    bin_normalizer[bin_idx + inc_x] += dx - dxdy;

    bin_idx += mask_size * inc_y;
    // (1 - dx) * dy = dy - dxdy
    bin_normalizer[bin_idx] += dy - dxdy;
    bin_normalizer[bin_idx + inc_x] += dxdy;
  }

  // Get normalization for each feature.
  float avg_normalizer = 0.0f;
  for (int k = 0; k < num_features; ++k) {
    const RegionFlowFeature& feature = feature_list.feature(k);
    const float grid_y = feature.y() * scale_y;
    const float grid_x = feature.x() * scale_x;

    const int int_grid_x = grid_x;
    const int int_grid_y = grid_y;

    const float dx = grid_x - int_grid_x;
    const float dy = grid_y - int_grid_y;
    const float dxdy = dx * dy;
    const float dx_plus_dy = dx + dy;

    int inc_x = dx != 0;
    int inc_y = dy != 0;

    float normalizer = 0;
    int bin_idx = int_grid_y * mask_size + int_grid_x;
    CHECK_LT(bin_idx, max_bins);
    // See above.
    normalizer += bin_normalizer[bin_idx] * (1 - dx_plus_dy + dxdy);
    normalizer += bin_normalizer[bin_idx + inc_x] * (dx - dxdy);

    bin_idx += mask_size * inc_y;
    CHECK_LT(bin_idx, max_bins);
    normalizer += bin_normalizer[bin_idx] * (dy - dxdy);
    normalizer += bin_normalizer[bin_idx + inc_x] * dxdy;

    const float inv_normalizer =
        normalizer > 0 ? 1.0f / std::sqrt(static_cast<double>(normalizer)) : 0;
    avg_normalizer += inv_normalizer;

    (*irls_weights)[k] *= inv_normalizer;
  }

  const float scale = num_features / (avg_normalizer + 1e-6f);

  // Normalize such that average scale is 1.0f.
  for (int k = 0; k < num_features; ++k) {
    (*irls_weights)[k] *= scale;
  }
}

void MotionEstimation::IrlsInitialization(
    const MotionType& type, int max_unstable_type, int frame,
    const EstimateModelOptions& model_options,
    SingleTrackClipData* clip_data) const {
  if (options_.estimation_policy() ==
      MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS) {
    CHECK_NE(frame, -1) << "Only per frame processing for this policy "
                        << "supported.";
  }

  IrlsInitializationInvoker invoker(type, max_unstable_type, model_options,
                                    this, clip_data);

  if (frame == -1) {
    // Function pointer to select between SerialFor and ParallelFor based on
    // EstimationPolicy.
    typedef void (*ForFunctionPtr)(size_t, size_t, size_t,
                                   const IrlsInitializationInvoker&);

    ForFunctionPtr for_function = &ParallelFor<IrlsInitializationInvoker>;

    // Inlier mask only used for translation or linear similarity.
    // In that case, initialization needs to proceed serially.
    if (type == MODEL_TRANSLATION || type == MODEL_LINEAR_SIMILARITY) {
      if (clip_data->inlier_mask != nullptr) {
        for_function = &SerialFor<IrlsInitializationInvoker>;
      }
    }

    for_function(0, clip_data->num_frames(), 1, invoker);
  } else {
    CHECK_GE(frame, 0);
    CHECK_LT(frame, clip_data->num_frames());
    invoker(BlockedRange(frame, frame + 1, 1));
  }
}

// Helper class for parallel irls weight filtering.
// This filters per-frame pair, across multiple models, not across time.
class TrackFilterInvoker {
 public:
  explicit TrackFilterInvoker(
      std::vector<MotionEstimation::SingleTrackClipData>* clip_datas)
      : clip_datas_(clip_datas) {}

  void operator()(const BlockedRange& range) const {
    for (int f = range.begin(); f != range.end(); ++f) {
      // Gather irls weights for each track.
      absl::node_hash_map<int, std::vector<float>> track_weights;
      for (auto& clip_data : *clip_datas_) {
        for (const auto& feature : (*clip_data.feature_lists)[f]->feature()) {
          track_weights[feature.track_id()].push_back(feature.irls_weight());
        }
      }

      // Min filter across weights, store in first element.
      int match_sum = 0;
      for (auto& entry : track_weights) {
        match_sum += entry.second.size();
        entry.second[0] =
            *std::min_element(entry.second.begin(), entry.second.end());
      }

      // Apply.
      for (auto& clip_data : *clip_datas_) {
        for (auto& feature :
             *(*clip_data.feature_lists)[f]->mutable_feature()) {
          feature.set_irls_weight(track_weights[feature.track_id()][0]);
        }
      }
    }
  }

 private:
  std::vector<MotionEstimation::SingleTrackClipData>* clip_datas_;
};

void MotionEstimation::MinFilterIrlsWeightByTrack(
    SingleTrackClipData* clip_data) const {
  // Gather irls weights for each track.
  absl::node_hash_map<int, std::vector<float>> track_weights;

  const int num_frames = clip_data->feature_lists->size();
  for (int f = 0; f < num_frames; ++f) {
    for (const auto& feature : clip_data->feature_lists->at(f)->feature()) {
      track_weights[feature.track_id()].push_back(feature.irls_weight());
    }
  }

  // Robust min filter across weights, store in first element; here 20th
  // percentile.
  for (auto& entry : track_weights) {
    if (entry.second.size() > 1) {
      auto robust_min =
          entry.second.begin() + std::ceil(entry.second.size() * 0.2f);

      std::nth_element(entry.second.begin(), robust_min, entry.second.end());
      entry.second[0] = *robust_min;
    }
  }

  // Apply.
  for (int f = 0; f < num_frames; ++f) {
    for (auto& feature : *clip_data->feature_lists->at(f)->mutable_feature()) {
      feature.set_irls_weight(track_weights[feature.track_id()][0]);
    }
  }
}

void MotionEstimation::EnforceTrackConsistency(
    std::vector<SingleTrackClipData>* clip_datas) const {
  CHECK(clip_datas != nullptr);
  if (clip_datas->empty()) {
    return;
  }

  // Traverse each frame, filter across clip_datas.
  const int num_frames = (*clip_datas)[0].num_frames();

  // Map track id to weights.
  SerialFor(0, num_frames, 1, TrackFilterInvoker(clip_datas));

  if (!options_.joint_track_estimation().temporal_smoothing()) {
    return;
  }

  // Temporal smoothing in time for each clip data.
  for (auto& clip_data : *clip_datas) {
    // Bilateral 1D filter across irls weights for each track.
    // Gather irls weights across all tracks.
    absl::node_hash_map<int, std::deque<float>> track_irls_weights;

    for (RegionFlowFeatureList* feature_list : *clip_data.feature_lists) {
      for (const auto& feature : feature_list->feature()) {
        track_irls_weights[feature.track_id()].push_back(feature.irls_weight());
      }
    }

    // Filter weights for each track, store results in map below.
    for (auto& iter : track_irls_weights) {
      SmoothIRLSWeights(&iter.second);
    }

    // Set filtered weight.
    for (RegionFlowFeatureList* feature_list : *clip_data.feature_lists) {
      for (auto& feature : *feature_list->mutable_feature()) {
        feature.set_irls_weight(track_irls_weights[feature.track_id()].front());
        track_irls_weights[feature.track_id()].pop_front();
      }
    }
  }
}

void MotionEstimation::BiasFromFeatures(
    const RegionFlowFeatureList& feature_list, MotionType type,
    const EstimateModelOptions& model_options, std::vector<float>* bias) const {
  CHECK(bias);
  const int num_features = feature_list.feature_size();
  bias->resize(num_features);

  int feature_idx = 0;
  const int type_idx = model_options.IndexFromType(type);
  auto& bias_map = long_feature_bias_maps_[type_idx];
  constexpr float kMinBias = 0.1f;

  for (const auto& feature : feature_list.feature()) {
    // Is feature present?
    auto iter = bias_map.find(feature.track_id());
    if (iter != bias_map.end()) {
      const float current_bias_bin =
          iter->second.bias * feature_bias_lut_.bias_weight_scale;

      // Never bias 100% towards old value,
      // allow for new values to propagate.
      // Downweight outliers but do not upweight inliers.
      if (current_bias_bin >= feature_bias_lut_.bias_weight_lut.size()) {
        (*bias)[feature_idx] = kMinBias;
      } else {
        (*bias)[feature_idx] = std::max(
            kMinBias, feature_bias_lut_.bias_weight_lut[current_bias_bin]);
      }
    } else {
      // TODO: This should be some kind of average of all the other
      // feature's bias; such that new features can't overpower old ones
      // (e.g. on tracking errors).
      (*bias)[feature_idx] = 1.0f;
    }

    ++feature_idx;
  }
}

void MotionEstimation::BiasLongFeatures(
    RegionFlowFeatureList* feature_list, MotionType type,
    const EstimateModelOptions& model_options,
    PriorFeatureWeights* prior_weights) const {
  CHECK(prior_weights);
  CHECK(feature_list);

  // Don't bias duplicated frames -> should be identity transform.
  if (feature_list->is_duplicated()) {
    return;
  }

  // TODO: Rename, it is not the bias (= error), but a weight in
  // [0, 1] to condition features.
  std::vector<float> bias;
  BiasFromFeatures(*feature_list, type, model_options, &bias);

  // Bias along long tracks.
  if (!prior_weights->use_full_prior) {
    LOG_IF(WARNING,
           []() {
             static int k = 0;
             return k++ < 2;
           }())
        << "Use full prior overridden to true, no initialization used. "
        << "Atypical usage.";
    prior_weights->use_full_prior = true;
  }

  const int num_features = feature_list->feature_size();
  if (prior_weights->priors.empty() && num_features > 0) {
    LOG(WARNING) << "BiasLongFeatures without using IrlsOutlierInitialization "
                 << "or LongFeatureInitialization.";
    prior_weights->priors.resize(num_features, 1.0f);
  }

  CHECK_EQ(num_features, prior_weights->priors.size());
  for (int k = 0; k < num_features; ++k) {
    prior_weights->priors[k] *= bias[k];
    auto* feature = feature_list->mutable_feature(k);
    feature->set_irls_weight(feature->irls_weight() * bias[k]);
  }
}

void MotionEstimation::ComputeSpatialBias(
    MotionType type, const EstimateModelOptions& model_options,
    RegionFlowFeatureList* feature_list, SpatialBiasMap* spatial_bias) const {
  const auto& bias_options = options_.long_feature_bias_options();
  const int type_idx = model_options.IndexFromType(type);
  const auto& bias_map = long_feature_bias_maps_[type_idx];

  // Select all features that are not marked to be ignored (irls weight of
  // zero).
  RegionFlowFeatureView feature_view;
  SelectFeaturesFromList(
      [](const RegionFlowFeature& feature) -> bool {
        return feature.irls_weight() != 0;
      },
      feature_list, &feature_view);

  const int num_features = feature_view.size();

  std::vector<std::vector<int>> feature_taps_3;
  std::vector<FeatureGrid<RegionFlowFeature>> feature_grids;

  // Create grid to seed newly observed features with an appropiate bias.
  BuildFeatureGrid(NormalizedDomain().x(), NormalizedDomain().y(),
                   bias_options.grid_size(), {feature_view}, FeatureLocation,
                   &feature_taps_3, nullptr, nullptr, &feature_grids);
  CHECK_EQ(1, feature_grids.size());
  const FeatureGrid<RegionFlowFeature>& single_grid = feature_grids[0];

  const float long_track_threshold = bias_options.long_track_threshold();

  // Traverse bins.
  for (int k = 0; k < single_grid.size(); ++k) {
    for (auto feature_ptr : single_grid[k]) {  // Traverse each bin.
      float total_weight = 0.0f;
      float weighted_bias = 0.0f;

      // Counts all neighbors considered (including itself).
      int total_neighbors = 0;

      // Counts approximately how many similar looking long feature tracks
      // are neighbors.
      float similar_long_tracks = 0;
      for (int neighbor_bin : feature_taps_3[k]) {
        for (auto neighbor_ptr : single_grid[neighbor_bin]) {
          ++total_neighbors;
          auto iter = bias_map.find(neighbor_ptr->track_id());
          float neighbor_bias = 0;
          int num_observations = 0;

          if (iter != bias_map.end()) {
            neighbor_bias = iter->second.bias;
            num_observations = iter->second.total_observations;
          } else {
            // If new track use estimated irls weight.
            neighbor_bias = 1.0f / neighbor_ptr->irls_weight();
            num_observations = 1;
          }

          const float distance =
              (FeatureLocation(*feature_ptr) - FeatureLocation(*neighbor_ptr))
                  .Norm();
          const float spatial_weight =
              feature_bias_lut_
                  .spatial_lut[distance * feature_bias_lut_.spatial_scale];

          const float color_distance =
              RegionFlowFeatureDistance(feature_ptr->feature_descriptor(),
                                        neighbor_ptr->feature_descriptor());

          const float color_weight =
              feature_bias_lut_
                  .color_lut[color_distance * feature_bias_lut_.color_scale];

          if (num_observations >= long_track_threshold) {
            // Count similar looking tracks (weights are normalized such that
            // identical looking track has color_weight of 1.0.
            // Scale by length of track (limited to kMaxTrackScale).
            constexpr float kMaxTrackScale = 3.0f;
            similar_long_tracks +=
                color_weight *
                std::min(kMaxTrackScale,
                         num_observations / long_track_threshold);
          }

          const float weight = spatial_weight * color_weight;

          total_weight += weight;
          weighted_bias += neighbor_bias * weight;
        }
      }

      DCHECK(spatial_bias->find(feature_ptr->track_id()) ==
             spatial_bias->end());

      // Threshold such that few similar tracks do not count.
      // Set to 0.25% of features.
      if (similar_long_tracks < 2.5e-3 * num_features) {
        similar_long_tracks = 0;
      }

      // Cutoff for features that do not have any matching neighbors.
      // In that case fallback to feature's irls weight.
      if (total_weight > total_neighbors * 1e-4f) {
        // Note: Considered doing minimum of bias and irls weight
        //       but this leads to instable IRLS weights very similar
        //       to independent estimation.
        const float norm_bias = weighted_bias / total_weight;
        (*spatial_bias)[feature_ptr->track_id()] =
            std::make_pair(norm_bias, similar_long_tracks);
      } else {
        (*spatial_bias)[feature_ptr->track_id()] = std::make_pair(
            1.0f / feature_ptr->irls_weight(), similar_long_tracks);
      }
    }
  }
}

void MotionEstimation::UpdateLongFeatureBias(
    MotionType type, const EstimateModelOptions& model_options,
    bool remove_terminated_tracks, bool update_irls_observation,
    RegionFlowFeatureList* feature_list) const {
  const int type_idx = model_options.IndexFromType(type);
  auto& bias_map = long_feature_bias_maps_[type_idx];

  constexpr int kMaxDuplicatedFrames = 2;
  int& model_duplicate_frames = num_duplicate_frames_[type_idx];
  if (feature_list->is_duplicated()) {
    ++model_duplicate_frames;
  } else {
    model_duplicate_frames = 0;
  }

  // Do not update bias from duplicated frames. We consider any duplicated
  // frames > kMaxDuplicatedFrames as a totally static camera, in which case the
  // update is desired.
  if (model_duplicate_frames > 0 &&
      model_duplicate_frames <= kMaxDuplicatedFrames) {
    for (auto& feature : *feature_list->mutable_feature()) {
      auto iter = bias_map.find(feature.track_id());
      if (iter != bias_map.end() && feature.irls_weight() > 0) {
        // Restore bias from last non-duplicated observation.
        feature.set_irls_weight(1.0f / (iter->second.bias + kIrlsEps));
      }
    }
    return;
  }

  const auto& bias_options = options_.long_feature_bias_options();
  const int num_irls_observations = bias_options.num_irls_observations();

  CHECK_GT(num_irls_observations, 0) << "Specify value > 0";
  const float inv_num_irls_observations = 1.0f / num_irls_observations;

  SpatialBiasMap spatial_bias;
  if (bias_options.use_spatial_bias()) {
    ComputeSpatialBias(type, model_options, feature_list, &spatial_bias);
  } else {
    // Just populate bias with inverse irls error.
    for (const auto& feature : feature_list->feature()) {
      spatial_bias[feature.track_id()] =
          std::make_pair(1.0f / feature.irls_weight(), 0);
    }
  }

  // Tracks current ids in this frame.
  std::unordered_set<int> curr_track_ids;

  // Scale applied to irls weight for linear interpolation between inlier and
  // outlier bias.
  CHECK_GT(bias_options.inlier_irls_weight(), 0);
  const float irls_scale = 1.0f / bias_options.inlier_irls_weight();
  const float long_track_scale =
      1.0f / bias_options.long_track_confidence_fraction();

  for (auto& feature : *feature_list->mutable_feature()) {
    if (remove_terminated_tracks) {
      curr_track_ids.insert(feature.track_id());
    }

    // Skip features that are marked as not to be processed.
    if (feature.irls_weight() == 0) {
      continue;
    }

    auto iter = bias_map.find(feature.track_id());

    // Is LongFeatureBias present?
    if (iter != bias_map.end()) {
      // Get minimum across last k observation.
      constexpr size_t lastK = 3;

      const std::vector<float>& irls_values = iter->second.irls_values;
      const float last_min = *std::min_element(
          irls_values.end() - std::min(irls_values.size(), lastK),
          irls_values.end());

      const float curr_irls_weight = feature.irls_weight();

      // Clamp weights for ratio computation (count major outliers and inliers
      // as regular ones). Set to range of errors between 0.5 pixels to 25
      // pixels.
      const float last_min_clamped = std::max(0.04f, std::min(last_min, 2.0f));

      const float curr_irls_clamped =
          std::max(0.04f, std::min(curr_irls_weight, 2.0f));

      // Only checking for change from outlier to inlier here.
      // The reverse case inlier -> outlier is addressed by bias
      // blending below. Denominator is guaranteed to be > 0.
      float irls_ratio = curr_irls_clamped / last_min_clamped;

      if (irls_ratio > bias_options.max_irls_change_ratio()) {
        // Reset feature and start again.
        bias_map[feature.track_id()] =
            LongFeatureBias(spatial_bias[feature.track_id()].first);
        continue;
      }

      ++iter->second.total_observations;

      // Compute median.
      std::vector<float> irls_values_copy = iter->second.irls_values;
      auto median = irls_values_copy.begin() + irls_values_copy.size() / 2;
      std::nth_element(irls_values_copy.begin(), median,
                       irls_values_copy.end());

      // By default shorter observations are given less prior, except if
      // sufficient long tracks were used during seeding.
      float prior_weight =
          std::max(std::min(1.0f, spatial_bias[feature.track_id()].second *
                                      long_track_scale),
                   (irls_values_copy.size() * inv_num_irls_observations));

      const float alpha = std::min(1.0f, *median * irls_scale) * prior_weight;
      const float bias = (alpha * bias_options.inlier_bias() +
                          (1.0f - alpha) * bias_options.outlier_bias());

      // Bias is weighted by median IRLS weight (foreground / outliers)
      // can override bias faster. Similar if only few IRLS values have
      // been observed, update bias faster.
      const float biased_weight =
          bias * iter->second.bias +
          (1.0f - bias) * (1.0f / feature.irls_weight());

      // Update weight.
      iter->second.bias = biased_weight;

      std::vector<float>& irls_values_ref = iter->second.irls_values;
      if (!update_irls_observation) {
        irls_values_ref.push_back(feature.irls_weight());
        if (irls_values_ref.size() > num_irls_observations) {
          irls_values_ref.erase(irls_values_ref.begin());
        }
      } else {
        // Update.
        // TODO: Sure about this? This is the error after
        // estimation, but also biased toward previous solution.
        irls_values_ref.back() = feature.irls_weight();
      }

      // Update feature's weight as well.
      feature.set_irls_weight(1.0f / (biased_weight + kIrlsEps));
    } else {
      CHECK(!update_irls_observation) << "Should never happen on >= 2nd round";

      // Not present, reset to spatial bias.
      const float biased_weight = spatial_bias[feature.track_id()].first;
      bias_map[feature.track_id()] = LongFeatureBias(biased_weight);
      feature.set_irls_weight(1.0f / (biased_weight + kIrlsEps));
    }
  }

  // Remove terminated tracks.
  if (remove_terminated_tracks) {
    std::vector<int> tracks_to_be_removed;
    for (const auto& entry : bias_map) {
      if (curr_track_ids.find(entry.first) == curr_track_ids.end()) {
        tracks_to_be_removed.push_back(entry.first);
      }
    }

    for (auto id : tracks_to_be_removed) {
      bias_map.erase(id);
    }
  }
}

void MotionEstimation::SmoothIRLSWeights(std::deque<float>* irls) const {
  CHECK(irls != nullptr);
  if (irls->empty()) {
    return;
  }

  const float sigma_space = 7.0f;
  const float sigma_signal = 0.5f;

  // Account for 90% of the data.
  const int radius = 1.65f * sigma_space + 0.5f;
  const int diameter = 2 * radius + 1;
  const int num_irls = irls->size();

  // Calculate spatial weights;
  std::vector<float> weights(diameter);
  const float space_coeff = -0.5f / (sigma_space * sigma_space);
  for (int i = -radius; i <= radius; ++i) {
    weights[i + radius] = std::exp(space_coeff * i * i);
  }

  // Map weights to error.
  std::vector<float> error(num_irls + 2 * radius);
  for (int k = 0; k < num_irls; ++k) {
    error[radius + k] = 1.0f / ((*irls)[k] + 1e-6f);
  }

  // Copy border (right hand side).
  std::copy(error.rbegin() + radius, error.rbegin() + 2 * radius,
            error.end() - radius);
  // Left hand side.
  std::copy(error.begin() + radius, error.begin() + 2 * radius, error.begin());

  // Bilateral filter.
  const float signal_coeff = (-0.5f / (sigma_signal * sigma_signal));
  for (int i = 0; i < num_irls; ++i) {
    const float curr_val = error[i + radius];
    float val_sum = 0;
    float weight_sum = 0;
    for (int k = 0; k < diameter; ++k) {
      const float value = error[i + k];
      const float diff = value - curr_val;
      const float weight =
          weights[k] *
          std::exp(static_cast<double>(diff * diff * signal_coeff));
      weight_sum += weight;
      val_sum += value * weight;
    }

    // Result is val_sum / weight_sum, as irls is inverse of error, result is:
    // weight_sum / val_sum.
    if (val_sum != 0) {
      (*irls)[i] = weight_sum / val_sum;
    }
  }
}

int MotionEstimation::IRLSRoundsFromSettings(const MotionType& type) const {
  const int irls_rounds = options_.irls_rounds();
  switch (type) {
    case MODEL_AVERAGE_MAGNITUDE:
      LOG(WARNING) << "Called with irls free motion type. Returning zero.";
      return 0;

    case MODEL_TRANSLATION:
      if (options_.estimate_translation_irls()) {
        return irls_rounds;
      } else {
        return 1;  // Always do at least L2 for translation.
      }
      break;

    case MODEL_LINEAR_SIMILARITY:
      switch (options_.linear_similarity_estimation()) {
        case MotionEstimationOptions::ESTIMATION_LS_NONE:
          return 0;

        case MotionEstimationOptions::ESTIMATION_LS_L2:
          return 1;

        case MotionEstimationOptions::ESTIMATION_LS_IRLS:
          return irls_rounds;

        case MotionEstimationOptions::ESTIMATION_LS_L2_RANSAC:
        case MotionEstimationOptions::ESTIMATION_LS_L1:
          LOG(FATAL) << "Deprecated options, use ESTIMATION_LS_IRLS instead.";
          return -1;
      }
      break;

    case MODEL_AFFINE:
      switch (options_.affine_estimation()) {
        case MotionEstimationOptions::ESTIMATION_AFFINE_NONE:
          return 0;

        case MotionEstimationOptions::ESTIMATION_AFFINE_L2:
          return 1;

        case MotionEstimationOptions::ESTIMATION_AFFINE_IRLS:
          return irls_rounds;
      }
      break;

    case MODEL_HOMOGRAPHY:
      switch (options_.homography_estimation()) {
        case MotionEstimationOptions::ESTIMATION_HOMOG_NONE:
          return 0;

        case MotionEstimationOptions::ESTIMATION_HOMOG_L2:
          return 1;

        case MotionEstimationOptions::ESTIMATION_HOMOG_IRLS:
          return irls_rounds;
      }
      break;

    case MODEL_MIXTURE_HOMOGRAPHY:
      switch (options_.mix_homography_estimation()) {
        case MotionEstimationOptions::ESTIMATION_HOMOG_MIX_NONE:
          return 0;

        case MotionEstimationOptions::ESTIMATION_HOMOG_MIX_L2:
          return 1;

        case MotionEstimationOptions::ESTIMATION_HOMOG_MIX_IRLS:
          return irls_rounds;
      }
      break;

    case MODEL_NUM_VALUES:
      LOG(FATAL) << "Function should never be called with this value";
      break;
  }

  LOG(FATAL) << "All branches above return, execution can not reach this point";
  return -1;
}

void MotionEstimation::PolicyToIRLSRounds(int irls_rounds, int* total_rounds,
                                          int* irls_per_round) const {
  CHECK(total_rounds != nullptr);
  CHECK(irls_per_round != nullptr);

  // Small optimization: irls_rounds == 0 -> total_rounds = 0 regardless of
  // settings.
  if (irls_rounds == 0) {
    *total_rounds = 0;
    *irls_per_round = 0;
    return;
  }

  switch (options_.estimation_policy()) {
    case MotionEstimationOptions::INDEPENDENT_PARALLEL:
    case MotionEstimationOptions::TEMPORAL_IRLS_MASK:
      *irls_per_round = irls_rounds;
      *total_rounds = 1;
      break;

    case MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS:
      *total_rounds = options_.long_feature_bias_options().total_rounds();
      *irls_per_round = irls_rounds;
      break;

    case MotionEstimationOptions::JOINTLY_FROM_TRACKS:
      *irls_per_round = 1;
      *total_rounds = irls_rounds;
      break;
  }
}

void MotionEstimation::CheckModelStability(
    const MotionType& type, const CameraMotion::Type& max_unstable_type,
    const std::vector<std::vector<float>>* reset_irls_weights,
    std::vector<RegionFlowFeatureList*>* feature_lists,
    std::vector<CameraMotion>* camera_motions) const {
  CHECK(feature_lists != nullptr);
  CHECK(camera_motions != nullptr);
  const int num_frames = feature_lists->size();
  if (reset_irls_weights) {
    DCHECK_EQ(num_frames, reset_irls_weights->size());
  }
  DCHECK_EQ(num_frames, camera_motions->size());

  for (int f = 0; f < num_frames; ++f) {
    CameraMotion& camera_motion = (*camera_motions)[f];
    RegionFlowFeatureList* feature_list = (*feature_lists)[f];

    const std::vector<float>* reset_irls_weight =
        reset_irls_weights ? &reset_irls_weights->at(f) : nullptr;
    CheckSingleModelStability(type, max_unstable_type, reset_irls_weight,
                              feature_list, &camera_motion);
  }
}

// Order of estimation is:
// Translation -> Linear Similarity -> Affine -> Homography -> Mixture
// Homography.
void MotionEstimation::CheckSingleModelStability(
    const MotionType& type, const CameraMotion::Type& max_unstable_type,
    const std::vector<float>* reset_irls_weights,
    RegionFlowFeatureList* feature_list, CameraMotion* camera_motion) const {
  if (camera_motion->type() > max_unstable_type) {
    return;
  }

  switch (type) {
    case MODEL_AVERAGE_MAGNITUDE:
      LOG(WARNING) << "Nothing to check for requested model type.";
      return;

    case MODEL_TRANSLATION:
      if (IsStableTranslation(camera_motion->translation(),
                              camera_motion->translation_variance(),
                              *feature_list)) {
        // Translation can never be singular.
        CHECK_EQ(
            0, camera_motion->flags() & CameraMotion::FLAG_SINGULAR_ESTIMATION);
      } else {
        // Invalid model.
        if (reset_irls_weights) {
          SetRegionFlowFeatureIRLSWeights(*reset_irls_weights, feature_list);
        }
        ResetMotionModels(options_, camera_motion);
      }
      break;

    case MODEL_LINEAR_SIMILARITY: {
      const int num_inliers =
          std::round(feature_list->feature_size() *
                     camera_motion->similarity_inlier_ratio());
      if (camera_motion->flags() & CameraMotion::FLAG_SINGULAR_ESTIMATION ||
          !IsStableSimilarity(camera_motion->linear_similarity(), *feature_list,
                              num_inliers)) {
        if (reset_irls_weights) {
          SetRegionFlowFeatureIRLSWeights(*reset_irls_weights, feature_list);
        }

        ResetToTranslation(camera_motion->translation(), camera_motion);
      }
      break;
    }

    case MODEL_AFFINE:
      // Not implemented, nothing to check here.
      break;

    case MODEL_HOMOGRAPHY:
      if (camera_motion->flags() & CameraMotion::FLAG_SINGULAR_ESTIMATION ||
          !IsStableHomography(camera_motion->homography(),
                              camera_motion->average_homography_error(),
                              camera_motion->homography_inlier_coverage())) {
        if (reset_irls_weights) {
          SetRegionFlowFeatureIRLSWeights(*reset_irls_weights, feature_list);
        }

        ResetToSimilarity(camera_motion->linear_similarity(), camera_motion);
      }
      break;

    case MODEL_MIXTURE_HOMOGRAPHY: {
      std::vector<float> block_coverage(
          camera_motion->mixture_inlier_coverage().begin(),
          camera_motion->mixture_inlier_coverage().end());
      const float mix_min_inlier_coverage =
          options_.stable_mixture_homography_bounds().min_inlier_coverage();

      if (camera_motion->flags() & CameraMotion::FLAG_SINGULAR_ESTIMATION ||
          !IsStableMixtureHomography(camera_motion->mixture_homography(),
                                     mix_min_inlier_coverage, block_coverage)) {
        // Unstable homography mixture.
        // Fall-back to previously estimated motion type.
        // Function is only called for CameraMotion::Type <= UNSTABLE.
        // In this case this means type is either UNSTABLE, UNSTABLE_SIM
        // or VALID.
        // (UNSTABLE_HOMOG flag is set by this function only during
        //  ResetToHomography below).
        switch (camera_motion->type()) {
          case CameraMotion::VALID:
            // Homography deemed stable, fallback to it.
            MotionEstimation::ResetToHomography(camera_motion->homography(),
                                                true,  // flag_as_unstable_model
                                                camera_motion);
            break;

          case CameraMotion::UNSTABLE_SIM:
            MotionEstimation::ResetToSimilarity(
                camera_motion->linear_similarity(), camera_motion);
            break;

          case CameraMotion::UNSTABLE:
            MotionEstimation::ResetToTranslation(camera_motion->translation(),
                                                 camera_motion);
            break;

          case CameraMotion::INVALID:
          case CameraMotion::UNSTABLE_HOMOG:
            LOG(FATAL) << "Unexpected CameraMotion::Type: "
                       << camera_motion->type();
            break;
        }

        if (reset_irls_weights) {
          SetRegionFlowFeatureIRLSWeights(*reset_irls_weights, feature_list);
        }

        // Clear rolling shutter guess in case it was computed.
        camera_motion->set_rolling_shutter_guess(-1);
        camera_motion->clear_mixture_homography_spectrum();
      } else {
        // Stable mixture homography can reset unstable type.
        camera_motion->set_overridden_type(camera_motion->type());
        camera_motion->set_type(CameraMotion::VALID);
        // Select weakest regularized mixture.
        camera_motion->set_rolling_shutter_motion_index(0);
      }

      break;
    }

    case MODEL_NUM_VALUES:
      LOG(FATAL) << "Function should not be called with this value";
      break;
  }
}

void MotionEstimation::ProjectMotionsDown(
    const MotionType& type, std::vector<CameraMotion>* camera_motions) const {
  CHECK(camera_motions != nullptr);
  for (auto& camera_motion : *camera_motions) {
    switch (type) {
      case MODEL_AVERAGE_MAGNITUDE:
      case MODEL_TRANSLATION:
      case MODEL_MIXTURE_HOMOGRAPHY:
      case MODEL_AFFINE:
        LOG(WARNING) << "Nothing to project for requested model type";
        return;

      case MODEL_HOMOGRAPHY:
        // Only project down if model was estimated, otherwise would propagate
        // identity.
        if (camera_motion.has_homography() &&
            camera_motion.type() <= CameraMotion::UNSTABLE_HOMOG) {
          LinearSimilarityModel lin_sim =
              *camera_motion.mutable_linear_similarity() =
                  AffineAdapter::ProjectToLinearSimilarity(
                      HomographyAdapter::ProjectToAffine(
                          camera_motion.homography(), frame_width_,
                          frame_height_),
                      frame_width_, frame_height_);
        }

        ABSL_FALLTHROUGH_INTENDED;

      case MODEL_LINEAR_SIMILARITY:
        // Only project down if model was estimated.
        if (camera_motion.has_linear_similarity() &&
            camera_motion.type() <= CameraMotion::UNSTABLE_SIM) {
          *camera_motion.mutable_translation() =
              LinearSimilarityAdapter::ProjectToTranslation(
                  camera_motion.linear_similarity(), frame_width_,
                  frame_height_);
        }
        break;

      case MODEL_NUM_VALUES:
        LOG(FATAL) << "Function should not be called with this value";
        break;
    }
  }
}

void MotionEstimation::IRLSWeightFilter(
    std::vector<RegionFlowFeatureList*>* feature_lists) const {
  CHECK(feature_lists != nullptr);
  for (auto feature_ptr : *feature_lists) {
    switch (options_.irls_weight_filter()) {
      case MotionEstimationOptions::IRLS_FILTER_TEXTURE:
        TextureFilteredRegionFlowFeatureIRLSWeights(
            0.5,  // Below texturedness threshold of 0.5
            1,    // Set irls_weight to 1.
            feature_ptr);
        break;

      case MotionEstimationOptions::IRLS_FILTER_CORNER_RESPONSE:
        CornerFilteredRegionFlowFeatureIRLSWeights(
            0.5,  // Below texturedness threshold of 0.5
            1,    // Set irls_weight to 1.
            feature_ptr);
        break;

      case MotionEstimationOptions::IRLS_FILTER_NONE:
        break;
    }
  }
}

void MotionEstimation::EstimateMotionsParallel(
    bool post_irls_weight_smoothing,
    std::vector<RegionFlowFeatureList*>* feature_lists,
    std::vector<CameraMotion>* camera_motions) const {
  CHECK(camera_motions != nullptr);
  camera_motions->clear();
  camera_motions->resize(feature_lists->size());

  // Normalize features.
  for (std::vector<RegionFlowFeatureList*>::iterator feature_list =
           feature_lists->begin();
       feature_list != feature_lists->end(); ++feature_list) {
    TransformRegionFlowFeatureList(normalization_transform_, *feature_list);
  }

  if (!options_.overlay_detection()) {
    EstimateMotionsParallelImpl(options_.irls_weights_preinitialized(),
                                feature_lists, camera_motions);
  } else {
    DetermineOverlayIndices(options_.irls_weights_preinitialized(),
                            camera_motions, feature_lists);

    EstimateMotionsParallelImpl(true, feature_lists, camera_motions);
  }

  if (!options_.deactivate_stable_motion_estimation()) {
    CheckTranslationAcceleration(camera_motions);
  }

  if (post_irls_weight_smoothing) {
    PostIRLSSmoothing(*camera_motions, feature_lists);
  }

  // Undo transform applied to features.
  for (auto& feature_list_ptr : *feature_lists) {
    TransformRegionFlowFeatureList(inv_normalization_transform_,
                                   feature_list_ptr);
  }

  DetermineShotBoundaries(*feature_lists, camera_motions);
}

void MotionEstimation::DetermineShotBoundaries(
    const std::vector<RegionFlowFeatureList*>& feature_lists,
    std::vector<CameraMotion>* camera_motions) const {
  CHECK(camera_motions != nullptr);
  CHECK_EQ(feature_lists.size(), camera_motions->size());
  const auto& shot_options = options_.shot_boundary_options();

  // Verify empty feature frames and invalid models via visual consistency.
  const int num_motions = camera_motions->size();
  for (int k = 0; k < num_motions; ++k) {
    auto& camera_motion = (*camera_motions)[k];
    if (camera_motion.type() == CameraMotion::INVALID ||
        feature_lists[k]->feature_size() == 0) {
      // Potential shot boundary, verify.
      if (feature_lists[k]->visual_consistency() >= 0) {
        if (feature_lists[k]->visual_consistency() >=
            shot_options.motion_consistency_threshold()) {
          camera_motion.set_flags(camera_motion.flags() |
                                  CameraMotion::FLAG_SHOT_BOUNDARY);
        }
      } else {
        // No consistency present, label as shot boundary.
        camera_motion.set_flags(camera_motion.flags() |
                                CameraMotion::FLAG_SHOT_BOUNDARY);
      }
    }
  }

  // Determine additional boundaries missed during motion estimation.
  for (int k = 0; k < num_motions; ++k) {
    auto& camera_motion = (*camera_motions)[k];
    if (feature_lists[k]->visual_consistency() >=
        shot_options.appearance_consistency_threshold()) {
      if (k + 1 == num_motions ||  // no next frame available.
          feature_lists[k + 1]->visual_consistency() >=
              shot_options.appearance_consistency_threshold()) {
        // Only add boundaries if previous or next frame are not already labeled
        // boundaries by above tests.
        if (k > 0 && ((*camera_motions)[k - 1].flags() &
                      CameraMotion::FLAG_SHOT_BOUNDARY) != 0) {
          continue;
        }
        if (k + 1 < num_motions && ((*camera_motions)[k + 1].flags() &
                                    CameraMotion::FLAG_SHOT_BOUNDARY) != 0) {
          continue;
        }

        // Shot boundaries based on visual consistency measure.
        camera_motion.set_flags(camera_motion.flags() |
                                CameraMotion::FLAG_SHOT_BOUNDARY);
      }
    }
  }

  // LOG shot boundaries.
  for (const auto& camera_motion : *camera_motions) {
    if (camera_motion.flags() & CameraMotion::FLAG_SHOT_BOUNDARY) {
      VLOG(1) << "Shot boundary at : " << camera_motion.timestamp_usec() * 1e-6f
              << "s";
    }
  }
}

void MotionEstimation::ResetMotionModels(const MotionEstimationOptions& options,
                                         CameraMotion* camera_motion) {
  CHECK(camera_motion);

  // Clear models.
  camera_motion->clear_translation();
  camera_motion->clear_similarity();
  camera_motion->clear_linear_similarity();
  camera_motion->clear_affine();
  camera_motion->clear_homography();
  camera_motion->clear_mixture_homography();
  camera_motion->clear_mixture_homography_spectrum();

  // We need to set models explicitly for has_* tests to work.
  *camera_motion->mutable_translation() = TranslationModel();

  if (options.estimate_similarity()) {
    *camera_motion->mutable_similarity() = SimilarityModel();
  }

  if (options.linear_similarity_estimation() !=
      MotionEstimationOptions::ESTIMATION_LS_NONE) {
    *camera_motion->mutable_linear_similarity() = LinearSimilarityModel();
  }

  if (options.affine_estimation() !=
      MotionEstimationOptions::ESTIMATION_AFFINE_NONE) {
    *camera_motion->mutable_affine() = AffineModel();
  }

  if (options.homography_estimation() !=
      MotionEstimationOptions::ESTIMATION_HOMOG_NONE) {
    *camera_motion->mutable_homography() = Homography();
  }

  if (options.mix_homography_estimation() !=
      MotionEstimationOptions::ESTIMATION_HOMOG_MIX_NONE) {
    *camera_motion->mutable_mixture_homography() =
        MixtureHomographyAdapter::IdentityModel(options.num_mixtures());
    camera_motion->set_mixture_row_sigma(options.mixture_row_sigma());
  }

  camera_motion->set_type(CameraMotion::INVALID);
}

void MotionEstimation::ResetToIdentity(CameraMotion* camera_motion,
                                       bool consider_valid) {
  if (camera_motion->has_translation()) {
    *camera_motion->mutable_translation() = TranslationModel();
  }

  if (camera_motion->has_similarity()) {
    *camera_motion->mutable_similarity() = SimilarityModel();
  }

  if (camera_motion->has_linear_similarity()) {
    *camera_motion->mutable_linear_similarity() = LinearSimilarityModel();
  }

  if (camera_motion->has_affine()) {
    *camera_motion->mutable_affine() = AffineModel();
  }

  if (camera_motion->has_homography()) {
    *camera_motion->mutable_homography() = Homography();
  }

  if (camera_motion->has_mixture_homography()) {
    const int num_models = camera_motion->mixture_homography().model_size();
    for (int m = 0; m < num_models; ++m) {
      *camera_motion->mutable_mixture_homography()->mutable_model(m) =
          Homography();
    }
  }

  if (consider_valid) {
    camera_motion->set_type(CameraMotion::VALID);
  } else {
    camera_motion->set_type(CameraMotion::INVALID);
  }
}

void MotionEstimation::ResetToTranslation(const TranslationModel& model,
                                          CameraMotion* camera_motion) {
  const float dx = model.dx();
  const float dy = model.dy();

  if (camera_motion->has_translation()) {
    *camera_motion->mutable_translation() = model;
  }

  if (camera_motion->has_similarity()) {
    *camera_motion->mutable_similarity() =
        SimilarityAdapter::FromArgs(dx, dy, 1, 0);
  }

  if (camera_motion->has_linear_similarity()) {
    *camera_motion->mutable_linear_similarity() =
        LinearSimilarityAdapter::FromArgs(dx, dy, 1, 0);
  }

  if (camera_motion->has_affine()) {
    *camera_motion->mutable_affine() = TranslationAdapter::ToAffine(model);
  }

  if (camera_motion->has_homography()) {
    *camera_motion->mutable_homography() =
        TranslationAdapter::ToHomography(model);
  }

  if (camera_motion->has_mixture_homography()) {
    const int num_models = camera_motion->mixture_homography().model_size();
    const Homography h = TranslationAdapter::ToHomography(model);
    for (int m = 0; m < num_models; ++m) {
      camera_motion->mutable_mixture_homography()->mutable_model(m)->CopyFrom(
          h);
    }
    camera_motion->mutable_mixture_homography()->set_dof(
        MixtureHomography::CONST_DOF);
  }

  camera_motion->set_type(CameraMotion::UNSTABLE);
}

void MotionEstimation::ResetToSimilarity(const LinearSimilarityModel& model,
                                         CameraMotion* camera_motion) {
  if (camera_motion->has_similarity()) {
    *camera_motion->mutable_similarity() =
        LinearSimilarityAdapter::ToSimilarity(model);
  }

  if (camera_motion->has_linear_similarity()) {
    *camera_motion->mutable_linear_similarity() = model;
  }

  if (camera_motion->has_affine()) {
    *camera_motion->mutable_affine() = LinearSimilarityAdapter::ToAffine(model);
  }

  if (camera_motion->has_homography()) {
    *camera_motion->mutable_homography() =
        LinearSimilarityAdapter::ToHomography(model);
  }

  if (camera_motion->has_mixture_homography()) {
    const int num_models = camera_motion->mixture_homography().model_size();
    const Homography h = LinearSimilarityAdapter::ToHomography(model);

    for (int m = 0; m < num_models; ++m) {
      camera_motion->mutable_mixture_homography()->mutable_model(m)->CopyFrom(
          h);
    }
    camera_motion->mutable_mixture_homography()->set_dof(
        MixtureHomography::CONST_DOF);
  }

  camera_motion->set_type(CameraMotion::UNSTABLE_SIM);
}

void MotionEstimation::ResetToHomography(const Homography& model,
                                         bool flag_as_unstable_model,
                                         CameraMotion* camera_motion) {
  if (camera_motion->has_homography()) {
    *camera_motion->mutable_homography() = model;
  }

  if (camera_motion->has_mixture_homography()) {
    const int num_models = camera_motion->mixture_homography().model_size();

    for (int m = 0; m < num_models; ++m) {
      camera_motion->mutable_mixture_homography()->mutable_model(m)->CopyFrom(
          model);
    }
    camera_motion->mutable_mixture_homography()->set_dof(
        MixtureHomography::CONST_DOF);
  }

  if (flag_as_unstable_model) {
    camera_motion->set_type(CameraMotion::UNSTABLE_HOMOG);
  }
}

void MotionEstimation::EstimateAverageMotionMagnitude(
    const RegionFlowFeatureList& feature_list,
    CameraMotion* camera_motion) const {
  std::vector<float> magnitudes;
  magnitudes.reserve(feature_list.feature_size());
  for (const auto& feature : feature_list.feature()) {
    magnitudes.push_back(std::hypot(feature.dy(), feature.dx()));
  }

  std::sort(magnitudes.begin(), magnitudes.end());
  auto tenth = magnitudes.begin() + magnitudes.size() / 10;
  auto ninetieth = magnitudes.begin() + magnitudes.size() * 9 / 10;
  const int elems = ninetieth - tenth;
  if (elems > 0) {
    const float average_magnitude =
        std::accumulate(tenth, ninetieth, 0.0f) * (1.0f / elems);

    // De-normalize translation.
    const float magnitude =
        LinearSimilarityAdapter::TransformPoint(inv_normalization_transform_,
                                                Vector2_f(average_magnitude, 0))
            .x();
    camera_motion->set_average_magnitude(magnitude);
  }
}

float MotionEstimation::IRLSPriorWeight(int iteration, int irls_rounds) const {
  // Iteration zero -> mapped to one
  // Iteration irls_rounds -> mapped to irls_prior_scale.
  return 1.0f - (iteration * (1.0f / irls_rounds) *
                 (1.0f - options_.irls_prior_scale()));
}

namespace {

// Returns weighted translational model from feature_list.
Vector2_f EstimateTranslationModelFloat(
    const RegionFlowFeatureList& feature_list) {
  Vector2_f mean_motion(0, 0);
  float weight_sum = 0;
  for (const auto& feature : feature_list.feature()) {
    mean_motion += FeatureFlow(feature) * feature.irls_weight();
    weight_sum += feature.irls_weight();
  }

  if (weight_sum > 0) {
    mean_motion *= (1.0f / weight_sum);
  }
  return mean_motion;
}

Vector2_f EstimateTranslationModelDouble(
    const RegionFlowFeatureList& feature_list) {
  Vector2_d mean_motion(0, 0);
  double weight_sum = 0;
  for (const auto& feature : feature_list.feature()) {
    mean_motion +=
        Vector2_d::Cast(FeatureFlow(feature)) * feature.irls_weight();
    weight_sum += feature.irls_weight();
  }

  if (weight_sum > 0) {
    mean_motion *= (1.0 / weight_sum);
  }

  return Vector2_f::Cast(mean_motion);
}

}  // namespace.

void MotionEstimation::ComputeFeatureMask(
    const RegionFlowFeatureList& feature_list, std::vector<int>* mask_indices,
    std::vector<float>* bin_normalizer) const {
  CHECK(mask_indices != nullptr);
  CHECK(bin_normalizer != nullptr);

  const int num_features = feature_list.feature_size();
  mask_indices->clear();
  mask_indices->reserve(num_features);

  const int mask_size = options_.feature_mask_size();
  const int max_bins = mask_size * mask_size;
  bin_normalizer->clear();
  bin_normalizer->resize(max_bins, 0.0f);

  const Vector2_f domain = NormalizedDomain();
  const float denom_x = 1.0f / domain.x();
  const float denom_y = 1.0f / domain.y();

  // Record index, but guard against out of bound error.
  for (const auto& feature : feature_list.feature()) {
    const int bin_idx = std::min<int>(
        max_bins,
        static_cast<int>(feature.y() * denom_y * mask_size) * mask_size +
            feature.x() * denom_x * mask_size);

    ++(*bin_normalizer)[bin_idx];
    mask_indices->push_back(bin_idx);
  }

  for (float& bin_value : *bin_normalizer) {
    bin_value = (bin_value == 0) ? 0 : sqrt(1.0 / bin_value);
  }
}

bool MotionEstimation::GetTranslationIrlsInitialization(
    RegionFlowFeatureList* feature_list,
    const EstimateModelOptions& model_options, float avg_camera_motion,
    InlierMask* inlier_mask, TranslationModel* best_model) const {
  CHECK(best_model != nullptr);

  const int num_features = feature_list->feature_size();
  if (!num_features) {
    return false;
  }

  // Bool indicator which features agree with model in each round.
  // In case no RANSAC rounds are performed considered all features inliers.
  std::vector<uint8> best_features(num_features, 1);
  std::vector<uint8> curr_features(num_features);
  float best_sum = 0;

  unsigned int seed = 900913;  // = Google in leet :)
  std::default_random_engine rand_gen(seed);
  std::uniform_int_distribution<> distribution(0, num_features - 1);

  auto& options = options_.irls_initialization();
  const float irls_residual_scale = GetIRLSResidualScale(
      avg_camera_motion, options_.irls_motion_magnitude_fraction());
  const float cutoff = options.cutoff() / irls_residual_scale;
  const float sq_cutoff = cutoff * cutoff;

  // Either temporal bias or motion prior based on options.
  std::vector<float> bias(num_features, 1.0f);

  // Optionally, compute mask index for each feature.
  std::vector<int> mask_indices;

  if (options_.estimation_policy() ==
      MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS) {
    BiasFromFeatures(*feature_list, MODEL_TRANSLATION, model_options, &bias);
  } else if (inlier_mask) {
    std::vector<float> unused_bin_normalizer;
    ComputeFeatureMask(*feature_list, &mask_indices, &unused_bin_normalizer);
    inlier_mask->MotionPrior(*feature_list, &bias);
  }

  for (int rounds = 0; rounds < options.rounds(); ++rounds) {
    float curr_sum = 0;
    // Pick a random vector.
    const int rand_idx = distribution(rand_gen);
    const Vector2_f flow = FeatureFlow(feature_list->feature(rand_idx));

    // curr_features gets set for every feature below; no need to reset.
    for (int i = 0; i < num_features; ++i) {
      const Feature& feature = feature_list->feature(i);
      const Vector2_f diff = FeatureFlow(feature) - flow;
      curr_features[i] = static_cast<uint8>(diff.Norm2() < sq_cutoff);
      if (curr_features[i]) {
        float score = feature.irls_weight();
        if (inlier_mask) {
          const int bin_idx = mask_indices[i];
          score *= bias[i] + inlier_mask->GetInlierScore(bin_idx);
        } else {
          score *= bias[i];
        }
        curr_sum += score;
      }
    }

    if (curr_sum > best_sum) {
      best_sum = curr_sum;
      best_features.swap(curr_features);
      best_model->set_dx(flow.x());
      best_model->set_dy(flow.y());
    }
  }

  if (inlier_mask) {
    inlier_mask->InitUpdateMask();
  }

  std::vector<float> inlier_weights;

  // Score outliers low.
  for (int i = 0; i < num_features; ++i) {
    RegionFlowFeature* feature = feature_list->mutable_feature(i);
    if (best_features[i] == 0 && feature->irls_weight() != 0) {
      feature->set_irls_weight(kOutlierIRLSWeight);
    } else {
      inlier_weights.push_back(feature->irls_weight());
      if (inlier_mask) {
        const int bin_idx = mask_indices[i];
        inlier_mask->RecordInlier(bin_idx, feature->irls_weight());
      }
    }
  }

  if (!inlier_weights.empty()) {
    // Ensure that all selected inlier features have at least median weight.
    auto median = inlier_weights.begin() + inlier_weights.size() * 0.5f;
    std::nth_element(inlier_weights.begin(), median, inlier_weights.end());

    for (int i = 0; i < num_features; ++i) {
      RegionFlowFeature* feature = feature_list->mutable_feature(i);
      if (best_features[i] != 0) {
        feature->set_irls_weight(std::max(*median, feature->irls_weight()));
      }
    }
  }

  // Compute translation variance as feature for stability evaluation.
  const float translation_variance = TranslationVariance(
      *feature_list, Vector2_f(best_model->dx(), best_model->dy()));

  return IsStableTranslation(*best_model, translation_variance, *feature_list);
}

void MotionEstimation::EstimateTranslationModelIRLS(
    int irls_rounds, bool compute_stability,
    RegionFlowFeatureList* flow_feature_list,
    const PriorFeatureWeights* prior_weights,
    CameraMotion* camera_motion) const {
  if (prior_weights && !prior_weights->HasCorrectDimension(
                           irls_rounds, flow_feature_list->feature_size())) {
    LOG(ERROR) << "Prior weights incorrectly initialized, ignoring.";
    prior_weights = nullptr;
  }

  // Simply average over features.
  const bool irls_use_l0_norm = options_.irls_use_l0_norm();

  const float irls_residual_scale =
      GetIRLSResidualScale(camera_motion->average_magnitude(),
                           options_.irls_motion_magnitude_fraction());

  const std::vector<float>* irls_priors = nullptr;
  const std::vector<float>* irls_alphas = nullptr;
  if (prior_weights && prior_weights->HasNonZeroAlpha()) {
    irls_priors = &prior_weights->priors;
    irls_alphas = &prior_weights->alphas;
  }

  Vector2_f mean_motion;
  for (int i = 0; i < irls_rounds; ++i) {
    if (options_.use_highest_accuracy_for_normal_equations()) {
      mean_motion = EstimateTranslationModelDouble(*flow_feature_list);
    } else {
      mean_motion = EstimateTranslationModelFloat(*flow_feature_list);
    }

    const float alpha = irls_alphas != nullptr ? (*irls_alphas)[i] : 0.0f;
    const float one_minus_alpha = 1.0f - alpha;

    // Update irls weights.
    const auto feature_start = flow_feature_list->mutable_feature()->begin();
    for (auto feature = feature_start;
         feature != flow_feature_list->mutable_feature()->end(); ++feature) {
      if (feature->irls_weight() == 0.0f) {
        continue;
      }

      // Express difference in original domain.
      const Vector2_f diff = LinearSimilarityAdapter::TransformPoint(
          irls_transform_, FeatureFlow(*feature) - mean_motion);

      const float numerator =
          alpha == 0.0f ? 1.0f
                        : ((*irls_priors)[feature - feature_start] * alpha +
                           one_minus_alpha);

      if (irls_use_l0_norm) {
        feature->set_irls_weight(
            numerator / (diff.Norm() * irls_residual_scale + kIrlsEps));
      } else {
        feature->set_irls_weight(
            numerator /
            (std::sqrt(static_cast<double>(diff.Norm() * irls_residual_scale)) +
             kIrlsEps));
      }
    }
  }

  // De-normalize translation.
  Vector2_f translation = LinearSimilarityAdapter::TransformPoint(
      inv_normalization_transform_, mean_motion);

  camera_motion->mutable_translation()->set_dx(translation.x());
  camera_motion->mutable_translation()->set_dy(translation.y());

  if (compute_stability) {
    camera_motion->set_translation_variance(
        TranslationVariance(*flow_feature_list, translation));
  }
}

float MotionEstimation::TranslationVariance(
    const RegionFlowFeatureList& feature_list,
    const Vector2_f& translation) const {
  // Compute irls based variance.
  float variance = 0;
  double weight_sum = 0;

  for (const auto& feature : feature_list.feature()) {
    weight_sum += feature.irls_weight();
    variance += (LinearSimilarityAdapter::TransformPoint(
                     inv_normalization_transform_, FeatureFlow(feature)) -
                 translation)
                    .Norm2() *
                feature.irls_weight();
  }

  if (weight_sum > 0) {
    return variance / weight_sum;
  } else {
    return 0.0f;
  }
}

namespace {

// Solves for the linear similarity via normal equations,
// using only the positions specified by features from the feature list.
// Input matrix is expected to be a 4x4 matrix of type T, rhs and solution are
// both 4x1 vectors of type T.
// Template class T specifies the desired accuracy, use float or double.
template <class T>
LinearSimilarityModel LinearSimilarityL2SolveSystem(
    const RegionFlowFeatureList& feature_list, Eigen::Matrix<T, 4, 4>* matrix,
    Eigen::Matrix<T, 4, 1>* rhs, Eigen::Matrix<T, 4, 1>* solution,
    bool* success) {
  CHECK(matrix != nullptr);
  CHECK(rhs != nullptr);
  CHECK(solution != nullptr);

  *matrix = Eigen::Matrix<T, 4, 4>::Zero();
  *rhs = Eigen::Matrix<T, 4, 1>::Zero();

  // Matrix multiplications are hand-coded for speed improvements vs.
  // opencv's cvGEMM calls.
  for (const auto& feature : feature_list.feature()) {
    const T x = feature.x();
    const T y = feature.y();
    const T w = feature.irls_weight();

    // double J[2 * 4] = {1, 0, x,  -y,
    //                    0, 1, y,   x};
    // Compute J^t * J * w = {1,  0,   x,    -y
    //                        0,  1,   y,     x,
    //                        x,  y,   xx+yy, 0,
    //                        -y  x,   0,     xx+yy} * w;

    const T x_w = x * w;
    const T y_w = y * w;
    const T xx_yy_w = (x * x + y * y) * w;

    T* matrix_ptr = matrix->data();
    matrix_ptr[0] += w;
    matrix_ptr[2] += x_w;
    matrix_ptr[3] += -y_w;

    matrix_ptr += 4;
    matrix_ptr[1] += w;
    matrix_ptr[2] += y_w;
    matrix_ptr[3] += x_w;

    matrix_ptr += 4;
    matrix_ptr[0] += x_w;
    matrix_ptr[1] += y_w;
    matrix_ptr[2] += xx_yy_w;

    matrix_ptr += 4;
    matrix_ptr[0] += -y_w;
    matrix_ptr[1] += x_w;
    matrix_ptr[3] += xx_yy_w;

    T* rhs_ptr = rhs->data();

    // Using identity parametrization below.
    const T m_x = feature.dx() * w;
    const T m_y = feature.dy() * w;

    rhs_ptr[0] += m_x;
    rhs_ptr[1] += m_y;
    rhs_ptr[2] += x * m_x + y * m_y;
    rhs_ptr[3] += -y * m_x + x * m_y;
  }

  // Solution parameters p.
  *solution = matrix->colPivHouseholderQr().solve(*rhs);
  if (((*matrix) * (*solution)).isApprox(*rhs, kPrecision)) {
    LinearSimilarityModel model;
    model.set_dx((*solution)(0, 0));
    model.set_dy((*solution)(1, 0));
    model.set_a((*solution)(2, 0) + 1.0);  // Identity parametrization.
    model.set_b((*solution)(3, 0));
    if (success) {
      *success = true;
    }
    return model;
  }

  if (success) {
    *success = false;
  }
  return LinearSimilarityModel();
}

}  // namespace.

bool MotionEstimation::GetSimilarityIrlsInitialization(
    RegionFlowFeatureList* feature_list,
    const EstimateModelOptions& model_options, float avg_camera_motion,
    InlierMask* inlier_mask, LinearSimilarityModel* best_model) const {
  CHECK(best_model != nullptr);

  const int num_features = feature_list->feature_size();
  if (!num_features) {
    return false;
  }

  // matrix is symmetric.
  Eigen::Matrix<float, 4, 4> matrix = Eigen::Matrix<float, 4, 4>::Zero();
  Eigen::Matrix<float, 4, 1> solution = Eigen::Matrix<float, 4, 1>::Zero();
  Eigen::Matrix<float, 4, 1> rhs = Eigen::Matrix<float, 4, 1>::Zero();

  // Bool indicator which features agree with model in each round.
  // In case no RANSAC rounds are performed considered all features inliers.
  std::vector<uint8> best_features(num_features, 1);
  std::vector<uint8> curr_features(num_features);
  float best_sum = 0;

  unsigned int seed = 900913;  // = Google in leet :)
  std::default_random_engine rand_gen(seed);
  std::uniform_int_distribution<> distribution(0, num_features - 1);
  auto& options = options_.irls_initialization();

  const float irls_residual_scale = GetIRLSResidualScale(
      avg_camera_motion, options_.irls_motion_magnitude_fraction());
  const float cutoff = options.cutoff() / irls_residual_scale;
  const float sq_cutoff = cutoff * cutoff;

  // Either temporal bias or motion prior based on options.
  std::vector<float> bias(num_features, 1.0f);

  // Compute mask index for each feature.
  std::vector<int> mask_indices;

  if (options_.estimation_policy() ==
      MotionEstimationOptions::TEMPORAL_LONG_FEATURE_BIAS) {
    BiasFromFeatures(*feature_list, MODEL_LINEAR_SIMILARITY, model_options,
                     &bias);
  } else if (inlier_mask) {
    std::vector<float> unused_bin_normalizer;
    ComputeFeatureMask(*feature_list, &mask_indices, &unused_bin_normalizer);
    inlier_mask->MotionPrior(*feature_list, &bias);
  }

  for (int rounds = 0; rounds < options.rounds(); ++rounds) {
    // Pick two random vectors.
    RegionFlowFeatureList to_test;
    to_test.add_feature()->CopyFrom(
        feature_list->feature(distribution(rand_gen)));
    to_test.add_feature()->CopyFrom(
        feature_list->feature(distribution(rand_gen)));
    ResetRegionFlowFeatureIRLSWeights(1.0f, &to_test);
    bool success = false;
    LinearSimilarityModel similarity = LinearSimilarityL2SolveSystem<float>(
        to_test, &matrix, &rhs, &solution, &success);
    if (!success) {
      continue;
    }

    float curr_sum = 0;
    for (int i = 0; i < num_features; ++i) {
      const Feature& feature = feature_list->feature(i);
      const Vector2_f trans_location = LinearSimilarityAdapter::TransformPoint(
          similarity, FeatureLocation(feature));
      const Vector2_f diff = FeatureMatchLocation(feature) - trans_location;
      curr_features[i] = static_cast<uint8>(diff.Norm2() < sq_cutoff);
      if (curr_features[i]) {
        float score = feature.irls_weight();
        if (inlier_mask) {
          const int bin_idx = mask_indices[i];
          score *= (bias[i] + inlier_mask->GetInlierScore(bin_idx));
        } else {
          score *= bias[i];
        }
        curr_sum += score;
      }
    }

    if (curr_sum > best_sum) {
      best_sum = curr_sum;
      best_features.swap(curr_features);
      best_model->Swap(&similarity);
    }
  }

  if (inlier_mask) {
    inlier_mask->InitUpdateMask();
  }

  int num_inliers = 0;

  std::vector<float> inlier_weights;

  // Score outliers low.
  for (int i = 0; i < num_features; ++i) {
    RegionFlowFeature* feature = feature_list->mutable_feature(i);
    if (best_features[i] == 0 && feature->irls_weight() != 0) {
      feature->set_irls_weight(kOutlierIRLSWeight);
    } else {
      ++num_inliers;
      inlier_weights.push_back(feature->irls_weight());
      if (inlier_mask) {
        const int bin_idx = mask_indices[i];
        inlier_mask->RecordInlier(bin_idx, feature->irls_weight());
      }
    }
  }

  if (!inlier_weights.empty()) {
    // Ensure that all selected inlier features have at least median weight.
    auto median = inlier_weights.begin() + inlier_weights.size() * 0.5f;
    std::nth_element(inlier_weights.begin(), median, inlier_weights.end());

    for (int i = 0; i < num_features; ++i) {
      RegionFlowFeature* feature = feature_list->mutable_feature(i);
      if (best_features[i] != 0) {
        feature->set_irls_weight(std::max(*median, feature->irls_weight()));
      }
    }
  }

  // For stability purposes we don't need to be that strict here.
  // Inflate number of actual inliers, as failing the initialization will most
  // likely fail the actual estimation. That way it can recover.
  num_inliers *= 2;
  return IsStableSimilarity(*best_model, *feature_list, num_inliers);
}

void MotionEstimation::ComputeSimilarityInliers(
    const RegionFlowFeatureList& feature_list, int* num_inliers,
    int* num_strict_inliers) const {
  CHECK(num_inliers);
  CHECK(num_strict_inliers);

  const auto& similarity_bounds = options_.stable_similarity_bounds();

  // Compute IRLS weight threshold from inlier threshold expressed in pixel
  // error. IRLS weight is normalized to 1 pixel error, so take reciprocal.
  float threshold = std::max<float>(similarity_bounds.inlier_threshold(),
                                    similarity_bounds.frac_inlier_threshold() *
                                        hypot(frame_width_, frame_height_));
  CHECK_GT(threshold, 0);

  threshold = 1.0f / threshold;
  float strict_threshold = similarity_bounds.strict_inlier_threshold();
  CHECK_GT(strict_threshold, 0);
  strict_threshold = 1.0f / strict_threshold;

  if (!options_.irls_use_l0_norm()) {
    threshold = std::sqrt(static_cast<double>(threshold));
  }

  *num_inliers = 0;
  *num_strict_inliers = 0;
  for (const auto& feature : feature_list.feature()) {
    if (feature.irls_weight() >= threshold) {
      ++*num_inliers;
    }

    if (feature.irls_weight() >= strict_threshold) {
      ++*num_strict_inliers;
    }
  }
}

bool MotionEstimation::EstimateLinearSimilarityModelIRLS(
    int irls_rounds, bool compute_stability,
    RegionFlowFeatureList* flow_feature_list,
    const PriorFeatureWeights* prior_weights,
    CameraMotion* camera_motion) const {
  if (prior_weights && !prior_weights->HasCorrectDimension(
                           irls_rounds, flow_feature_list->feature_size())) {
    LOG(ERROR) << "Prior weights incorrectly initialized, ignoring.";
    prior_weights = nullptr;
  }

  // Just declaring does not actually allocate memory
  Eigen::Matrix<float, 4, 4> matrix_f;
  Eigen::Matrix<float, 4, 1> solution_f;
  Eigen::Matrix<float, 4, 1> rhs_f;
  Eigen::Matrix<double, 4, 4> matrix_d;
  Eigen::Matrix<double, 4, 1> solution_d;
  Eigen::Matrix<double, 4, 1> rhs_d;

  if (options_.use_highest_accuracy_for_normal_equations()) {
    matrix_d = Eigen::Matrix<double, 4, 4>::Zero();
    solution_d = Eigen::Matrix<double, 4, 1>::Zero();
    rhs_d = Eigen::Matrix<double, 4, 1>::Zero();
  } else {
    matrix_f = Eigen::Matrix<float, 4, 4>::Zero();
    solution_f = Eigen::Matrix<float, 4, 1>::Zero();
    rhs_f = Eigen::Matrix<float, 4, 1>::Zero();
  }

  LinearSimilarityModel* solved_model =
      camera_motion->mutable_linear_similarity();

  const float irls_residual_scale =
      GetIRLSResidualScale(camera_motion->average_magnitude(),
                           options_.irls_motion_magnitude_fraction());

  const bool irls_use_l0_norm = options_.irls_use_l0_norm();

  const std::vector<float>* irls_priors = nullptr;
  const std::vector<float>* irls_alphas = nullptr;
  if (prior_weights && prior_weights->HasNonZeroAlpha()) {
    irls_priors = &prior_weights->priors;
    irls_alphas = &prior_weights->alphas;
  }

  for (int i = 0; i < irls_rounds; ++i) {
    bool success;
    if (options_.use_highest_accuracy_for_normal_equations()) {
      *solved_model = LinearSimilarityL2SolveSystem<double>(
          *flow_feature_list, &matrix_d, &rhs_d, &solution_d, &success);
    } else {
      *solved_model = LinearSimilarityL2SolveSystem<float>(
          *flow_feature_list, &matrix_f, &rhs_f, &solution_f, &success);
    }

    if (!success) {
      VLOG(1) << "Linear similarity estimation failed.";
      *camera_motion->mutable_linear_similarity() = LinearSimilarityModel();
      camera_motion->set_flags(camera_motion->flags() |
                               CameraMotion::FLAG_SINGULAR_ESTIMATION);
      return false;
    }

    const float alpha = irls_alphas != nullptr ? (*irls_alphas)[i] : 0.0f;
    const float one_minus_alpha = 1.0f - alpha;

    const auto feature_start = flow_feature_list->mutable_feature()->begin();
    for (auto feature = feature_start;
         feature != flow_feature_list->mutable_feature()->end(); ++feature) {
      if (feature->irls_weight() == 0.0f) {
        continue;
      }

      const Vector2_f trans_location = LinearSimilarityAdapter::TransformPoint(
          *solved_model, FeatureLocation(*feature));
      const Vector2_f matched_location = FeatureMatchLocation(*feature);

      // Express residual in frame coordinates.
      const Vector2_f residual = LinearSimilarityAdapter::TransformPoint(
          irls_transform_, trans_location - matched_location);
      const float numerator =
          alpha == 0.0f ? 1.0f
                        : ((*irls_priors)[feature - feature_start] * alpha +
                           one_minus_alpha);

      if (irls_use_l0_norm) {
        feature->set_irls_weight(
            numerator / (residual.Norm() * irls_residual_scale + kIrlsEps));
      } else {
        feature->set_irls_weight(
            numerator / (std::sqrt(static_cast<double>(residual.Norm() *
                                                       irls_residual_scale)) +
                         kIrlsEps));
      }
    }
  }

  // Undo pre_transform.
  *solved_model = ModelCompose3(inv_normalization_transform_, *solved_model,
                                normalization_transform_);

  if (compute_stability) {
    int num_inliers = 0;
    int num_strict_inliers = 0;

    if (flow_feature_list->feature_size() > 0) {
      ComputeSimilarityInliers(*flow_feature_list, &num_inliers,
                               &num_strict_inliers);

      const float inv_num_feat = 1.0f / flow_feature_list->feature_size();
      camera_motion->set_similarity_inlier_ratio(num_inliers * inv_num_feat);
      camera_motion->set_similarity_strict_inlier_ratio(num_strict_inliers *
                                                        inv_num_feat);
    } else {
      camera_motion->set_similarity_inlier_ratio(1.0f);
      camera_motion->set_similarity_strict_inlier_ratio(1.0f);
    }
  }

  return true;
}

bool MotionEstimation::EstimateAffineModelIRLS(
    int irls_rounds, RegionFlowFeatureList* feature_list,
    CameraMotion* camera_motion) const {
  // Setup solution matrices in column major.
  Eigen::Matrix<double, 6, 6> matrix = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> rhs = Eigen::Matrix<double, 6, 1>::Zero();

  AffineModel* solved_model = camera_motion->mutable_affine();

  // Multiple rounds of weighting based L2 optimization.
  for (int i = 0; i < irls_rounds; ++i) {
    // Build Jacobians.
    for (const auto& feature : feature_list->feature()) {
      const double w = feature.irls_weight();
      const Vector2_f& pt_1 = FeatureLocation(feature);
      const double x = pt_1.x() * w;
      const double y = pt_1.y() * w;

      Eigen::Matrix<double, 2, 6> jacobian =
          Eigen::Matrix<double, 2, 6>::Zero();
      jacobian(0, 0) = w;
      jacobian(0, 2) = x;
      jacobian(0, 3) = y;
      jacobian(1, 1) = w;
      jacobian(1, 4) = x;
      jacobian(1, 5) = y;

      // Update A.
      // Compute J^t * J, where J = jacobian.
      matrix = jacobian.transpose() * jacobian + matrix;

      // Transform matched point.
      const Vector2_f& pt_2 = FeatureMatchLocation(feature);
      Eigen::Matrix<double, 2, 1> pt_2_mat(pt_2.x() * w, pt_2.y() * w);

      // Compute J^t * y_i;
      rhs = jacobian.transpose() * pt_2_mat + rhs;
    }

    // Solve A * p = b;
    Eigen::Matrix<double, 6, 1> p = Eigen::Matrix<double, 6, 1>::Zero();
    p = matrix.colPivHouseholderQr().solve(rhs);
    if (!(matrix * p).isApprox(rhs, kPrecision)) {
      camera_motion->set_flags(camera_motion->flags() |
                               CameraMotion::FLAG_SINGULAR_ESTIMATION);
      return false;
    }

    // Set model.
    solved_model->set_dx(p(0, 0));
    solved_model->set_dy(p(1, 0));
    solved_model->set_a(p(2, 0));
    solved_model->set_b(p(3, 0));
    solved_model->set_c(p(4, 0));
    solved_model->set_d(p(5, 0));

    // Re-compute weights from errors.
    for (auto& feature : *feature_list->mutable_feature()) {
      if (feature.irls_weight() == 0.0f) {
        continue;
      }

      const Vector2_f trans_location = AffineAdapter::TransformPoint(
          *solved_model, FeatureLocation(feature));
      const Vector2_f& matched_location = FeatureMatchLocation(feature);

      // Express residual in frame coordinates.
      const Vector2_f residual = LinearSimilarityAdapter::TransformPoint(
          irls_transform_, trans_location - matched_location);

      feature.set_irls_weight(sqrt(1.0 / (residual.Norm() + kIrlsEps)));
    }
  }

  // Express in original frame coordinate system.
  *solved_model = ModelCompose3(
      LinearSimilarityAdapter::ToAffine(inv_normalization_transform_),
      *solved_model,
      LinearSimilarityAdapter::ToAffine(normalization_transform_));
  return true;
}

// Estimates homography via least squares (specifically QR decomposition).
// Specifically, for
// H = (a  b  t1)
//      c  d  t2
//      w1 w2  1
// H * (x, y, 1)^T = (a*x + b*y + t1
//                    c*x + d*y + t2
//                    w1*x + w2*y + 1 )
//
// Therefore matching point (mx, my) is given by
// mx = (a*x + b*y + t1) / (w1*x + w2*y + 1)     Eq. 1
// my = (c*x + d*y + t2) / (w1*x + w2*y + 1)
//
// Multiply with denominator to get:
// a*x + b*y + t1 -w1*x*mx -w2*y*mx = mx         Eq. 2
// c*x + d*y + t2 -w1*x*my -w2*y*my = my
//
// This results in the following system
// J = (x  y  1  0  0  0  -x*mx  -y*mx) * (a b t1 c d t2 w1 w2)^T  = (mx
//      0  0  0  x  y  1  -x*my  -y*my                                my)
//
// Note, that in a linear system Eq. 1 and Eq. 2 are not equivalent,
// as the denominator varies w.r.t. the position of each feature.

// Therefore, if the optional argument prev_solution is passed,
// each J will be scaled with the denominator w1*x + w2*y + 1.
//
// If perspective_regularizer := r != 0, an additional equation is introduced:
// (0 0 0 0 0 0 r r) * (a b t1 c d t2 w1 w2)^T = 0.
//
// Returns false if system could not be solved for.
template <class T>
bool HomographyL2QRSolve(
    const RegionFlowFeatureList& feature_list,
    const Homography* prev_solution,  // optional.
    float perspective_regularizer,
    Eigen::Matrix<T, Eigen::Dynamic, 8>* matrix,  // tmp matrix
    Eigen::Matrix<T, 8, 1>* solution) {
  CHECK(matrix);
  CHECK(solution);
  CHECK_EQ(8, matrix->cols());
  const int num_rows =
      2 * feature_list.feature_size() + (perspective_regularizer == 0 ? 0 : 1);
  CHECK_EQ(num_rows, matrix->rows());
  CHECK_EQ(1, solution->cols());
  CHECK_EQ(8, solution->rows());

  // Compute homography from features (H * location = prev_location).
  *matrix = Eigen::Matrix<T, Eigen::Dynamic, 8>::Zero(matrix->rows(), 8);
  Eigen::Matrix<T, Eigen::Dynamic, 1> rhs =
      Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(matrix->rows(), 1);

  if (RegionFlowFeatureIRLSSum(feature_list) > kMaxCondition) {
    return false;
  }

  // Create matrix and rhs (using h_33 = 1 constraint).
  int feature_idx = 0;
  for (auto feature = feature_list.feature().begin();
       feature != feature_list.feature().end(); ++feature, ++feature_idx) {
    int feature_row = 2 * feature_idx;

    Vector2_f pt = FeatureLocation(*feature);
    Vector2_f prev_pt = FeatureMatchLocation(*feature);
    // Weight per feature.
    double scale = 1.0;
    if (prev_solution) {
      const double denom =
          prev_solution->h_20() * pt.x() + prev_solution->h_21() * pt.y() + 1.0;
      if (fabs(denom) > 1e-5) {
        scale /= denom;
      } else {
        scale = 0;
      }
    }

    const float w = feature->irls_weight() * scale;

    // Scale feature with weight;
    Vector2_f pt_w = pt * w;

    // Row 1 of above J:
    (*matrix)(feature_row, 0) = pt_w.x();
    (*matrix)(feature_row, 1) = pt_w.y();
    (*matrix)(feature_row, 2) = w;

    // Entry 3 .. 5 equal zero.
    (*matrix)(feature_row, 6) = -pt_w.x() * prev_pt.x();
    (*matrix)(feature_row, 7) = -pt_w.y() * prev_pt.x();
    rhs(feature_row, 0) = prev_pt.x() * w;

    // Row 2 of above J:
    // Entry 0 .. 2 equal zero.
    (*matrix)(feature_row + 1, 3) = pt_w.x();
    (*matrix)(feature_row + 1, 4) = pt_w.y();
    (*matrix)(feature_row + 1, 5) = w;

    (*matrix)(feature_row + 1, 6) = -pt_w.x() * prev_pt.y();
    (*matrix)(feature_row + 1, 7) = -pt_w.y() * prev_pt.y();
    rhs(feature_row + 1, 0) = prev_pt.y() * w;
  }

  if (perspective_regularizer > 0) {
    int last_row_idx = 2 * feature_list.feature_size();
    (*matrix)(last_row_idx, 6) = (*matrix)(last_row_idx, 7) =
        perspective_regularizer;
  }

  // TODO: Consider a faster function?
  *solution = matrix->colPivHouseholderQr().solve(rhs);
  return ((*matrix) * (*solution)).isApprox(rhs, kPrecision);
}

// Same as function above, but solves for homography via normal equations,
// using only the positions specified by features from the feature list.
// Expects 8x8 matrix of type T and 8x1 rhs and solution vector of type T.
// Optional parameter is prev_solution, in which case each row is scaled by
// correct denominator (see derivation at function description
// HomographyL2QRSolve).
// Template class T specifies the desired accuracy, use float or double.
template <class T>
Homography HomographyL2NormalEquationSolve(
    const RegionFlowFeatureList& feature_list,
    const Homography* prev_solution,  // optional.
    float perspective_regularizer, Eigen::Matrix<T, 8, 8>* matrix,
    Eigen::Matrix<T, 8, 1>* rhs, Eigen::Matrix<T, 8, 1>* solution,
    bool* success) {
  CHECK(matrix != nullptr);
  CHECK(rhs != nullptr);
  CHECK(solution != nullptr);

  *matrix = Eigen::Matrix<T, 8, 8>::Zero();
  *rhs = Eigen::Matrix<T, 8, 1>::Zero();

  // Matrix multiplications are hand-coded for speed improvements vs.
  // opencv's cvGEMM calls.
  for (const auto& feature : feature_list.feature()) {
    T scale = 1.0;
    if (prev_solution) {
      const T denom = prev_solution->h_20() * feature.x() +
                      prev_solution->h_21() * feature.y() + 1.0;
      if (fabs(denom) > 1e-5) {
        scale /= denom;
      } else {
        scale = 0;
      }
    }
    const T w = feature.irls_weight() * scale;
    const T x = feature.x();
    const T y = feature.y();
    const T xw = x * w;
    const T yw = y * w;
    const T xxw = x * x * w;
    const T yyw = y * y * w;
    const T xyw = x * y * w;
    const T mx = feature.x() + feature.dx();
    const T my = feature.y() + feature.dy();

    const T mxxyy = mx * mx + my * my;
    // Jacobian
    // double J[2 * 8] = {x, y, 1,  0,  0,   0, -x * m_x, -y * m_x,
    //                   {0, 0, 0,  x,  y,   1, -x * m_y, -y * m_y}
    //
    // // Compute J^t * J * w =
    // ( xx        xy    x      0       0    0    -xx*mx  -xy*mx    )
    // ( xy        yy    y      0       0    0    -xy*mx  -yy*mx    )
    // ( x         y     1      0       0    0     -x*mx   -y*mx    )
    // ( 0         0     0     xx      xy    x    -xx*my  -xy*my    )
    // ( 0         0     0     xy      yy    y    -xy*my  -yy*my    )
    // ( 0         0     0      x      y     1     -x*my   -y*my    )
    // ( -xx*mx -xy*mx -x*mx -xx*my -xy*my -x*my xx*mxxyy  xy*mxxyy )
    // ( -xy*mx -yy*mx -y*mx -xy*my -yy*my -y*my xy*mxxyy  yy*mxxyy  ) * w

    // 1st row:  xx        xy    x      0       0    0    -xx*mx  -xy*mx
    T* matrix_ptr = matrix->data();
    matrix_ptr[0] += xxw;
    matrix_ptr[1] += xyw;
    matrix_ptr[2] += xw;
    matrix_ptr[6] += -xxw * mx;
    matrix_ptr[7] += -xyw * mx;

    // 2nd row:  xy       yy   y      0       0    0    -xy*mx  -yy*mx
    matrix_ptr += 8;
    matrix_ptr[0] += xyw;
    matrix_ptr[1] += yyw;
    matrix_ptr[2] += yw;
    matrix_ptr[6] += -xyw * mx;
    matrix_ptr[7] += -yyw * mx;

    // 3rd row: x         y     1      0       0    0     -x*mx   -y*mx
    matrix_ptr += 8;
    matrix_ptr[0] += xw;
    matrix_ptr[1] += yw;
    matrix_ptr[2] += w;
    matrix_ptr[6] += -xw * mx;
    matrix_ptr[7] += -yw * mx;

    // 4th row: 0         0     0     xx      xy    x    -xx*my  -xy*my
    matrix_ptr += 8;
    matrix_ptr[3] += xxw;
    matrix_ptr[4] += xyw;
    matrix_ptr[5] += xw;
    matrix_ptr[6] += -xxw * my;
    matrix_ptr[7] += -xyw * my;

    // 5th row: 0         0     0     xy      yy    y    -xy*my  -yy*my
    matrix_ptr += 8;
    matrix_ptr[3] += xyw;
    matrix_ptr[4] += yyw;
    matrix_ptr[5] += yw;
    matrix_ptr[6] += -xyw * my;
    matrix_ptr[7] += -yyw * my;

    // 6th row:  0         0     0     x     y     1      -x*my    -y*my
    matrix_ptr += 8;
    matrix_ptr[3] += xw;
    matrix_ptr[4] += yw;
    matrix_ptr[5] += w;
    matrix_ptr[6] += -xw * my;
    matrix_ptr[7] += -yw * my;

    // 7th row:  -xx*mx -xy*mx -x*mx -xx*my -xy*my -x*my xx*mxxyy  xy*mxxyy
    matrix_ptr += 8;
    matrix_ptr[0] += -xxw * mx;
    matrix_ptr[1] += -xyw * mx;
    matrix_ptr[2] += -xw * mx;
    matrix_ptr[3] += -xxw * my;
    matrix_ptr[4] += -xyw * my;
    matrix_ptr[5] += -xw * my;
    matrix_ptr[6] += xxw * mxxyy;
    matrix_ptr[7] += xyw * mxxyy;

    // 8th row: -xy*mx -yy*mx -y*mx -xy*my -yy*my -y*my xy*mxxyy  yy*mxxyy
    matrix_ptr += 8;
    matrix_ptr[0] += -xyw * mx;
    matrix_ptr[1] += -yyw * mx;
    matrix_ptr[2] += -yw * mx;
    matrix_ptr[3] += -xyw * my;
    matrix_ptr[4] += -yyw * my;
    matrix_ptr[5] += -yw * my;
    matrix_ptr[6] += xyw * mxxyy;
    matrix_ptr[7] += yyw * mxxyy;

    // Right hand side:
    // b = ( x
    //       y )
    // Compute J^t * b  * w =
    // ( x*mx  y*mx  mx  x*my  y*my  my  -x*mxxyy -y*mxxyy ) * w
    T* rhs_ptr = rhs->data();
    rhs_ptr[0] += xw * mx;
    rhs_ptr[1] += yw * mx;
    rhs_ptr[2] += mx * w;
    rhs_ptr[3] += xw * my;
    rhs_ptr[4] += yw * my;
    rhs_ptr[5] += my * w;
    rhs_ptr[6] += -xw * mxxyy;
    rhs_ptr[7] += -yw * mxxyy;
  }

  if (perspective_regularizer > 0) {
    // Additional constraint:
    // C[8] = {0, 0, 0,  0,  0,  0, r, r}
    // Compute C^t * C =
    // [ 0  ...       0   0  0
    //      ...
    //   0  ...       0   r^2  r^2
    //   0  ...       0   r^2  r^2 ]
    const T sq_r = perspective_regularizer * perspective_regularizer;

    T* matrix_ptr = matrix->row(6).data();
    matrix_ptr[6] += sq_r;
    matrix_ptr[7] += sq_r;
    matrix_ptr += 8;
    matrix_ptr[6] += sq_r;
    matrix_ptr[7] += sq_r;
    // Nothing to add to RHS (zero).
  }

  // Solution parameters p.
  *solution = matrix->colPivHouseholderQr().solve(*rhs);
  if (((*matrix) * (*solution)).isApprox(*rhs, kPrecision)) {
    const T* ptr = solution->data();
    Homography model;

    model.set_h_00(ptr[0]);
    model.set_h_01(ptr[1]);
    model.set_h_02(ptr[2]);
    model.set_h_10(ptr[3]);
    model.set_h_11(ptr[4]);
    model.set_h_12(ptr[5]);
    model.set_h_20(ptr[6]);
    model.set_h_21(ptr[7]);

    if (success) {
      *success = true;
    }
    return model;
  }

  if (success) {
    *success = false;
  }
  return Homography();
}

namespace {

float PatchDescriptorIRLSWeight(const RegionFlowFeature& feature) {
  float weight = feature.irls_weight();

  // Blend weight to combine irls weight with a feature's path standard
  // deviation.
  const float alpha = 0.7f;
  // Inverse of maximum value of standard deviation for intensities in [0, 255].
  // Scaled such that only low textured regions are given small weight.
  const float denom = 1.0f / 128.0f * 5.0f;

  const float feature_stdev_l1 =
      PatchDescriptorColorStdevL1(feature.feature_descriptor());

  if (feature_stdev_l1 >= 0.0f) {
    weight *= alpha + (1.f - alpha) * std::min(1.f, feature_stdev_l1 * denom);
  }

  return weight;
}

// Extension of above function to evenly spaced row-mixture models.
bool MixtureHomographyL2DLTSolve(
    const RegionFlowFeatureList& feature_list, int num_models,
    const MixtureRowWeights& row_weights, float regularizer_lambda,
    Eigen::MatrixXf* matrix,  // least squares matrix
    Eigen::MatrixXf* solution) {
  CHECK(matrix);
  CHECK(solution);

  // cv::solve can hang for really bad conditioned systems.
  const double feature_irls_sum = RegionFlowFeatureIRLSSum(feature_list);
  if (feature_irls_sum > kMaxCondition) {
    return false;
  }

  const int num_dof = 8 * num_models;
  const int num_constraints = num_dof - 8;

  CHECK_EQ(matrix->cols(), num_dof);
  // 2 Rows (x,y) per feature.
  CHECK_EQ(matrix->rows(), 2 * feature_list.feature_size() + num_constraints);
  CHECK_EQ(solution->cols(), 1);
  CHECK_EQ(solution->rows(), num_dof);

  // Compute homography from features. (H * location = prev_location)
  *matrix = Eigen::MatrixXf::Zero(matrix->rows(), matrix->cols());
  Eigen::Matrix<float, Eigen::Dynamic, 1> rhs =
      Eigen::MatrixXf::Zero(matrix->rows(), 1);

  // Normalize feature sum to 1.
  float irls_denom = 1.0 / (feature_irls_sum + 1e-6);

  // Create matrix for DLT.
  int feature_idx = 0;
  for (auto feature = feature_list.feature().begin();
       feature != feature_list.feature().end(); ++feature, ++feature_idx) {
    float* mat_row_1 = matrix->row(2 * feature_idx).data();
    float* mat_row_2 = matrix->row(2 * feature_idx + 1).data();
    float* rhs_row_1 = rhs.row(2 * feature_idx).data();
    float* rhs_row_2 = rhs.row(2 * feature_idx + 1).data();

    Vector2_f pt = FeatureLocation(*feature);
    Vector2_f prev_pt = FeatureMatchLocation(*feature);
    // Weight per feature.
    const float f_w = PatchDescriptorIRLSWeight(*feature) * irls_denom;

    // Scale feature point by weight;
    Vector2_f pt_w = pt * f_w;
    const float* mix_weights = row_weights.RowWeightsClamped(feature->y());

    for (int m = 0; m < num_models; ++m, mat_row_1 += 8, mat_row_2 += 8) {
      const float w = mix_weights[m];
      // Entries 0 .. 2 are zero.
      mat_row_1[3] = -pt_w.x() * w;
      mat_row_1[4] = -pt_w.y() * w;
      mat_row_1[5] = -f_w * w;

      mat_row_1[6] = pt_w.x() * prev_pt.y() * w;
      mat_row_1[7] = pt_w.y() * prev_pt.y() * w;

      mat_row_2[0] = pt_w.x() * w;
      mat_row_2[1] = pt_w.y() * w;
      mat_row_2[2] = f_w * w;

      // Entries 3 .. 5 are zero.
      mat_row_2[6] = -pt_w.x() * prev_pt.x() * w;
      mat_row_2[7] = -pt_w.y() * prev_pt.x() * w;
    }

    // Weights sum to one (-> take out of loop).
    rhs_row_1[0] = -prev_pt.y() * f_w;
    rhs_row_2[0] = prev_pt.x() * f_w;
  }

  // Add regularizer term. It is important to weight perspective larger
  // to roughly obtain similar magnitudes across parameters.
  const float param_weights[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 100.f, 100.f};

  const int reg_row_start = 2 * feature_list.feature_size();
  for (int m = 0; m < num_models - 1; ++m) {
    for (int p = 0; p < 8; ++p) {
      const int curr_idx = m * 8 + p;
      float* curr_row = matrix->row(reg_row_start + curr_idx).data();
      curr_row[curr_idx] = regularizer_lambda * param_weights[p];
      curr_row[curr_idx + 8] = -regularizer_lambda * param_weights[p];
    }
  }

  // TODO: Consider a faster function?
  *solution = matrix->colPivHouseholderQr().solve(rhs);
  return ((*matrix) * (*solution)).isApprox(rhs, kPrecision);
}

// Constraint mixture homography model.
// Only translation (2 DOF) varies across mixture of size num_models, with
// strictly affine and perspective part (4 + 2 = 6 DOF) being constant across
// the mixtures.
bool TransMixtureHomographyL2DLTSolve(
    const RegionFlowFeatureList& feature_list, int num_models,
    const MixtureRowWeights& row_weights, float regularizer_lambda,
    Eigen::MatrixXf* matrix,  // least squares matrix
    Eigen::MatrixXf* solution) {
  CHECK(matrix);
  CHECK(solution);

  // cv::solve can hang for really bad conditioned systems.
  const double feature_irls_sum = RegionFlowFeatureIRLSSum(feature_list);
  if (feature_irls_sum > kMaxCondition) {
    return false;
  }

  const int num_dof = 6 + 2 * num_models;
  const int num_constraints = 2 * (num_models - 1);

  CHECK_EQ(matrix->cols(), num_dof);
  // 2 Rows (x,y) per feature.
  CHECK_EQ(matrix->rows(), 2 * feature_list.feature_size() + num_constraints);
  CHECK_EQ(solution->cols(), 1);
  CHECK_EQ(solution->rows(), num_dof);

  // Compute homography from features. (H * location = prev_location)
  *matrix = Eigen::MatrixXf::Zero(matrix->rows(), matrix->cols());
  Eigen::Matrix<float, Eigen::Dynamic, 1> rhs =
      Eigen::MatrixXf::Zero(matrix->rows(), 1);

  // Create matrix for DLT.
  int feature_idx = 0;

  // Normalize feature sum to 1.
  float irls_denom = 1.0 / (feature_irls_sum + 1e-6);

  for (auto feature = feature_list.feature().begin();
       feature != feature_list.feature().end(); ++feature, ++feature_idx) {
    float* mat_row_1 = matrix->row(2 * feature_idx).data();
    float* mat_row_2 = matrix->row(2 * feature_idx + 1).data();
    float* rhs_row_1 = rhs.row(2 * feature_idx).data();
    float* rhs_row_2 = rhs.row(2 * feature_idx + 1).data();

    Vector2_f pt = FeatureLocation(*feature);
    Vector2_f prev_pt = FeatureMatchLocation(*feature);

    // Weight per feature.
    const float f_w = PatchDescriptorIRLSWeight(*feature) * irls_denom;

    // Scale feature point by weight.
    Vector2_f pt_w = pt * f_w;
    const float* mix_weights = row_weights.RowWeightsClamped(feature->y());

    // Entries 0 .. 1 are zero.
    mat_row_1[2] = -pt_w.x();
    mat_row_1[3] = -pt_w.y();

    mat_row_1[4] = pt_w.x() * prev_pt.y();
    mat_row_1[5] = pt_w.y() * prev_pt.y();

    mat_row_2[0] = pt_w.x();
    mat_row_2[1] = pt_w.y();

    // Entries 2 .. 3 are zero.
    mat_row_2[4] = -pt_w.x() * prev_pt.x();
    mat_row_2[5] = -pt_w.y() * prev_pt.x();

    // Weights sum to one (-> take out of loop).
    rhs_row_1[0] = -prev_pt.y() * f_w;
    rhs_row_2[0] = prev_pt.x() * f_w;

    for (int m = 0; m < num_models; ++m, mat_row_1 += 2, mat_row_2 += 2) {
      const float w = mix_weights[m];
      mat_row_1[6] = 0;
      mat_row_1[7] = -f_w * w;

      mat_row_2[6] = f_w * w;
      mat_row_2[7] = 0;
    }
  }

  const int reg_row_start = 2 * feature_list.feature_size();
  int constraint_idx = 0;
  for (int m = 0; m < num_models - 1; ++m) {
    for (int p = 0; p < 2; ++p, ++constraint_idx) {
      const int curr_idx = 6 + m * 2 + p;
      float* curr_row = matrix->row(reg_row_start + constraint_idx).data();
      curr_row[curr_idx] = regularizer_lambda;
      curr_row[curr_idx + 2] = -regularizer_lambda;
    }
  }

  // TODO: Consider a faster function
  *solution = matrix->colPivHouseholderQr().solve(rhs);
  return ((*matrix) * (*solution)).isApprox(rhs, kPrecision);
}

// Constraint mixture homography model.
// Only translation and skew and rotation (2 + 2 = 4 DOF) vary across mixture
// of size num_models, with scale and perspective part (2 + 2 = 4 DOF) being
// constant across the mixtures.
bool SkewRotMixtureHomographyL2DLTSolve(
    const RegionFlowFeatureList& feature_list, int num_models,
    const MixtureRowWeights& row_weights, float regularizer_lambda,
    Eigen::MatrixXf* matrix,  // least squares matrix
    Eigen::MatrixXf* solution) {
  CHECK(matrix);
  CHECK(solution);

  // cv::solve can hang for really bad conditioned systems.
  const double feature_irls_sum = RegionFlowFeatureIRLSSum(feature_list);
  if (feature_irls_sum > kMaxCondition) {
    return false;
  }

  const int num_dof = 4 + 4 * num_models;
  const int num_constraints = 4 * (num_models - 1);

  CHECK_EQ(matrix->cols(), num_dof);
  // 2 Rows (x,y) per feature.
  CHECK_EQ(matrix->rows(), 2 * feature_list.feature_size() + num_constraints);
  CHECK_EQ(solution->cols(), 1);
  CHECK_EQ(solution->rows(), num_dof);

  // Compute homography from features. (H * location = prev_location)
  *matrix = Eigen::MatrixXf::Zero(matrix->rows(), matrix->cols());
  Eigen::Matrix<float, Eigen::Dynamic, 1> rhs =
      Eigen::MatrixXf::Zero(matrix->rows(), 1);

  // Create matrix for DLT.
  int feature_idx = 0;

  // Normalize feature sum to 1.
  float irls_denom = 1.0 / (feature_irls_sum + 1e-6);

  for (auto feature = feature_list.feature().begin();
       feature != feature_list.feature().end(); ++feature, ++feature_idx) {
    Vector2_f pt = FeatureLocation(*feature);
    Vector2_f prev_pt = FeatureMatchLocation(*feature);

    // Weight per feature.
    const float f_w = PatchDescriptorIRLSWeight(*feature) * irls_denom;

    // Scale feature point by weight.
    Vector2_f pt_w = pt * f_w;
    const float* mix_weights = row_weights.RowWeightsClamped(feature->y());

    // Compare to MixtureHomographyDLTSolve.
    // Mapping of parameters (from homography to mixture) is as follows:
    //       0 1 2 3 4 5 6 7
    //  -->  0 4 6 5 1 7 2 3

    int feature_row = 2 * feature_idx;
    // Entry 0 is zero.
    // Skew is in mixture.
    (*matrix)(feature_row, 1) = -pt_w.y();
    (*matrix)(feature_row, 2) = pt_w.x() * prev_pt.y();
    (*matrix)(feature_row, 3) = pt_w.y() * prev_pt.y();

    (*matrix)(feature_row + 1, 0) = pt_w.x();
    // Entry 1 is zero.
    (*matrix)(feature_row + 1, 2) = -pt_w.x() * prev_pt.x();
    (*matrix)(feature_row + 1, 3) = -pt_w.y() * prev_pt.x();

    // Weights sum to one (-> take out of loop).
    rhs(feature_row, 0) = -prev_pt.y() * f_w;
    rhs(feature_row + 1, 0) = prev_pt.x() * f_w;

    // Is this right?
    for (int m = 0; m < num_models; ++m) {
      const float w = mix_weights[m];
      (*matrix)(feature_row, 4 + 4 * m) = 0.0f;
      (*matrix)(feature_row, 5 + 4 * m) = -pt_w.x() * w;  // Skew.
      (*matrix)(feature_row, 6 + 4 * m) = 0.0f;
      (*matrix)(feature_row, 7 + 4 * m) = -f_w * w;

      (*matrix)(feature_row + 1, 4 + 4 * m) = pt_w.y() * w;
      (*matrix)(feature_row + 1, 5 + 4 * m) = 0.0f;  // Translation.
      (*matrix)(feature_row + 1, 6 + 4 * m) = f_w * w;
      (*matrix)(feature_row + 1, 7 + 4 * m) = 0.0f;
    }
  }

  const int reg_row_start = 2 * feature_list.feature_size();
  int constraint_idx = 0;
  for (int m = 0; m < num_models - 1; ++m) {
    for (int p = 0; p < 4; ++p, ++constraint_idx) {
      const int curr_idx = 4 + m * 4 + p;
      int curr_row = reg_row_start + constraint_idx;
      (*matrix)(curr_row, curr_idx) = regularizer_lambda;
      (*matrix)(curr_row, curr_idx + 4) = -regularizer_lambda;
    }
  }

  // TODO: Consider a faster function?
  *solution = matrix->colPivHouseholderQr().solve(rhs);
  return ((*matrix) * (*solution)).isApprox(rhs, kPrecision);
}

}  // namespace.

// For plot example for IRLS_WEIGHT_PERIMITER_GAUSSIAN, see: goo.gl/fNzQc
// (plot assumes 3:2 ratio for width:height).
void MotionEstimation::GetHomographyIRLSCenterWeights(
    const RegionFlowFeatureList& feature_list,
    std::vector<float>* weights) const {
  CHECK(weights != nullptr);

  const int num_features = feature_list.feature_size();
  weights->clear();
  weights->reserve(num_features);

  // Early return for constant weight.
  if (options_.homography_irls_weight_initialization() ==
      MotionEstimationOptions::IRLS_WEIGHT_CONSTANT_ONE) {
    weights->resize(num_features, 1.0f);
    return;
  }

  const float sigma_x = normalized_domain_.x() * 0.3f;
  const float sigma_y = normalized_domain_.y() * 0.3f;
  const float denom_x = 1.0f / (sigma_x * sigma_x);
  const float denom_y = 1.0f / (sigma_y * sigma_y);
  const Vector2_f center = normalized_domain_ * 0.5f;

  for (const auto& feature : feature_list.feature()) {
    const float diff_x = feature.x() - center.x();
    const float diff_y = feature.y() - center.y();
    const float dist = diff_x * diff_x * denom_x + diff_y * diff_y * denom_y;
    const float weight = std::exp(static_cast<double>(-0.5f * dist));

    switch (options_.homography_irls_weight_initialization()) {
      case MotionEstimationOptions::IRLS_WEIGHT_CENTER_GAUSSIAN:
        weights->push_back(weight);
        break;
      case MotionEstimationOptions::IRLS_WEIGHT_PERIMETER_GAUSSIAN:
        weights->push_back(1.0f - weight * 0.5f);
        break;
      default:
        LOG(INFO) << "Unsupported IRLS weighting.";
    }
  }
}

bool MotionEstimation::IsStableTranslation(
    const TranslationModel& translation, float translation_variance,
    const RegionFlowFeatureList& features) const {
  if (options_.deactivate_stable_motion_estimation()) {
    return true;
  }

  const bool sufficient_features =
      features.feature_size() >=
      options_.stable_translation_bounds().min_features();

  if (!sufficient_features) {
    VLOG(1) << "Translation unstable, insufficient features.";
    return false;
  }

  const float translation_magnitude =
      std::hypot(translation.dx(), translation.dy());
  const float max_translation_magnitude =
      options_.stable_translation_bounds().frac_max_motion_magnitude() *
      hypot(frame_width_, frame_height_);

  const float stdev = std::sqrt(static_cast<double>(translation_variance)) /
                      hypot(frame_width_, frame_height_);

  const float max_motion_stdev_threshold =
      options_.stable_translation_bounds().max_motion_stdev_threshold();

  // Only test for exceeding max translation magnitude if standard deviation
  // of translation is not close to zero (in which case registration
  // can be considered stable).
  if (translation_magnitude >= max_translation_magnitude &&
      stdev >= max_motion_stdev_threshold) {
    VLOG(1) << "Translation unstable, exceeds max translation: "
            << translation_magnitude << " stdev: " << stdev;
    return false;
  }

  if (stdev >= options_.stable_translation_bounds().max_motion_stdev()) {
    VLOG(1) << "Translation unstable, translation variance out of bound: "
            << stdev;
    return false;
  }

  return true;
}

void MotionEstimation::CheckTranslationAcceleration(
    std::vector<CameraMotion>* camera_motions) const {
  CHECK(camera_motions != nullptr);
  std::vector<float> magnitudes;
  for (const auto& motion : *camera_motions) {
    const float translation_magnitude =
        LinearSimilarityAdapter::TransformPoint(
            normalization_transform_,
            Vector2_f(motion.translation().dx(), motion.translation().dy()))
            .Norm();

    magnitudes.push_back(translation_magnitude);
  }

  // Determine median motion around each frame
  // (really third lower percentile here).
  const int median_radius = 6;
  const int num_magnitudes = magnitudes.size();
  std::vector<float> median_magnitudes;
  const float kZeroMotion = 3e-4f;  // 0.5 pixels @ 720p.
  for (int k = 0; k < num_magnitudes; ++k) {
    std::vector<float> filter;
    const auto mag_begin =
        magnitudes.begin() + std::max(0, (k - median_radius));
    const auto mag_end =
        magnitudes.begin() + std::min(num_magnitudes, k + median_radius + 1);

    for (auto mag = mag_begin; mag != mag_end; ++mag) {
      // Ignore zero motion (empty or duplicate frames).
      if (*mag > kZeroMotion) {
        filter.push_back(*mag);
      }
    }

    const float kMinMotion = 1e-3f;  // 1.5 pixels @ 720p.
    if (filter.empty()) {
      median_magnitudes.push_back(kMinMotion);
    } else {
      auto median_iter = filter.begin() + filter.size() / 3;
      std::nth_element(filter.begin(), median_iter, filter.end());
      median_magnitudes.push_back(std::max(kMinMotion, *median_iter));
    }
  }

  const float max_acceleration =
      options_.stable_translation_bounds().max_acceleration();
  for (int k = 0; k < magnitudes.size(); ++k) {
    // Relative test, test for acceleration and de-acceleration (only
    // in case motion is not zero, to ignore empty or duplicated frames).
    if (magnitudes[k] > max_acceleration * median_magnitudes[k] ||
        (magnitudes[k] > kZeroMotion &&
         median_magnitudes[k] > max_acceleration * magnitudes[k])) {
      MotionEstimation::ResetMotionModels(options_, &(*camera_motions)[k]);
    }
  }
}

bool MotionEstimation::IsStableSimilarity(
    const LinearSimilarityModel& model,
    const RegionFlowFeatureList& feature_list, int num_inliers) const {
  if (options_.deactivate_stable_motion_estimation()) {
    // Only check if model is invertible.
    return IsInverseStable(model);
  }

  const auto& similarity_bounds = options_.stable_similarity_bounds();

  if (similarity_bounds.only_stable_input() && feature_list.unstable()) {
    VLOG(1) << "Feature list is unstable.";
    return false;
  }

  const float lower_scale = similarity_bounds.lower_scale();
  const float upper_scale = similarity_bounds.upper_scale();
  if (model.a() < lower_scale || model.a() > upper_scale) {
    VLOG(1) << "Warning: Unstable similarity found. "
            << "Scale is out of bound: " << model.a();
    return false;
  }

  const float limit_rotation = similarity_bounds.limit_rotation();
  if (fabs(model.b()) > limit_rotation) {
    VLOG(1) << "Warning: Unstable similarity found. "
            << "Rotation is out of bound: " << model.b();
    return false;
  }

  if (num_inliers < similarity_bounds.min_inliers()) {
    VLOG(1) << "Unstable similarity, only " << num_inliers << " inliers chosen "
            << "from " << feature_list.feature_size() << " features.";
    return false;
  }

  if (num_inliers <
      similarity_bounds.min_inlier_fraction() * feature_list.feature_size()) {
    VLOG(1) << "Unstable similarity, inlier fraction only "
            << static_cast<float>(num_inliers) /
                   (feature_list.feature_size() + 1.e-6f);
    return false;
  }

  return true;
}

bool MotionEstimation::IsStableHomography(const Homography& model,
                                          float average_homography_error,
                                          float inlier_coverage) const {
  if (options_.deactivate_stable_motion_estimation()) {
    return IsInverseStable(model);
  }

  const auto& homography_bounds = options_.stable_homography_bounds();

  // Test if the inter-frame transform is stable, i.e. small enough
  // that its faithful estimation is possible.
  const float lower_scale = homography_bounds.lower_scale();
  const float upper_scale = homography_bounds.upper_scale();
  if (model.h_00() < lower_scale || model.h_00() > upper_scale ||  // Scale.
      model.h_11() < lower_scale || model.h_11() > upper_scale) {
    VLOG(1) << "Warning: Unstable homography found. "
            << "Scale is out of bound: " << model.h_00() << " " << model.h_11();
    return false;
  }

  const float limit_rotation = homography_bounds.limit_rotation();
  if (fabs(model.h_01()) > limit_rotation ||  // Rotation.
      fabs(model.h_10()) > limit_rotation) {
    VLOG(1) << "Warning: Unstable homography found. "
            << "Rotation is out of bound: " << model.h_01() << " "
            << model.h_10();
    return false;
  }

  const float limit_perspective = homography_bounds.limit_perspective();
  if (fabs(model.h_20()) > limit_perspective ||  // Perspective.
      fabs(model.h_21()) > limit_perspective) {
    VLOG(1) << "Warning: Unstable homography found. "
            << "Perspective is out of bound:" << model.h_20() << " "
            << model.h_21();
    return false;
  }

  const float min_inlier_coverage = homography_bounds.min_inlier_coverage();
  const float registration_threshold =
      std::max<float>(homography_bounds.registration_threshold(),
                      homography_bounds.frac_registration_threshold() *
                          hypot(frame_width_, frame_height_));

  if (average_homography_error > registration_threshold &&
      inlier_coverage <= min_inlier_coverage) {
    VLOG(1) << "Unstable homography found. "
            << "Registration (actual, threshold): " << average_homography_error
            << " " << registration_threshold
            << " Inlier coverage (actual, threshold): " << inlier_coverage
            << " " << min_inlier_coverage;
    return false;
  }

  return true;
}

bool MotionEstimation::IsStableMixtureHomography(
    const MixtureHomography& homography, float min_block_inlier_coverage,
    const std::vector<float>& block_inlier_coverage) const {
  if (options_.deactivate_stable_motion_estimation()) {
    return true;
  }

  const int num_blocks = block_inlier_coverage.size();
  std::vector<bool> stable_block(num_blocks, false);
  for (int k = 0; k < num_blocks; ++k) {
    stable_block[k] = block_inlier_coverage[k] > min_block_inlier_coverage;
  }

  int unstable_start = -1;
  int empty_start = -1;
  const int max_outlier_blocks =
      options_.stable_mixture_homography_bounds().max_adjacent_outlier_blocks();
  const int max_empty_blocks =
      options_.stable_mixture_homography_bounds().max_adjacent_empty_blocks();

  for (int k = 0; k < num_blocks; ++k) {
    const int offset = unstable_start == 0 ? 1 : 0;
    // Test for outlier blocks.
    if (stable_block[k]) {
      if (unstable_start >= 0 &&
          k - unstable_start >= max_outlier_blocks - offset) {
        return false;
      }
      unstable_start = -1;
    } else {
      if (unstable_start < 0) {
        unstable_start = k;
      }
    }

    // Test for empty blocks.
    if (block_inlier_coverage[k] > 0) {
      if (empty_start >= 0 && k - empty_start >= max_empty_blocks - offset) {
        return false;
      }
      empty_start = -1;
    } else {
      if (empty_start < 0) {
        empty_start = k;
      }
    }
  }

  if (unstable_start >= 0 &&
      num_blocks - unstable_start >= max_outlier_blocks) {
    return false;
  }

  if (empty_start >= 0 && num_blocks - empty_start >= max_empty_blocks) {
    return false;
  }

  return true;
}

float MotionEstimation::GridCoverage(
    const RegionFlowFeatureList& feature_list, float min_inlier_score,
    MotionEstimationThreadStorage* thread_storage) const {
  CHECK(thread_storage != nullptr);

  // 10x10 grid for coverage estimation.
  const int grid_size = options_.coverage_grid_size();
  const int mask_size = grid_size * grid_size;

  const float scaled_width = 1.0f / normalized_domain_.x() * grid_size;
  const float scaled_height = 1.0f / normalized_domain_.y() * grid_size;

  const std::vector<float>& grid_cell_weights =
      thread_storage->GridCoverageInitializationWeights();
  CHECK_EQ(mask_size, grid_cell_weights.size());

  const float max_inlier_score = 1.75f * min_inlier_score;
  const float mid_inlier_score = 0.5 * (min_inlier_score + max_inlier_score);

  // Map min_inlier to 0.1 and max_inlier to 0.9 via logistic regression.
  // f(x) = 1 / (1 + exp(-a(x - mid)))
  // f(min) == 0.1 ==> a = ln(1 / 0.1 - 1) / (mid - min)
  const float logistic_scale = 2.1972245 /  // ln(1.0 / 0.1 - 1)
                               (mid_inlier_score - min_inlier_score);

  const int num_overlaps = 3;

  // Maximum coverage and number of features across shifted versions.
  std::vector<float> max_coverage(mask_size, 0.0f);
  std::vector<int> max_features(mask_size, 0);

  for (int overlap_y = 0; overlap_y < num_overlaps; ++overlap_y) {
    const float shift_y =
        normalized_domain_.y() / grid_size * overlap_y / num_overlaps;
    for (int overlap_x = 0; overlap_x < num_overlaps; ++overlap_x) {
      const float shift_x =
          normalized_domain_.x() / grid_size * overlap_x / num_overlaps;
      std::vector<std::vector<float>>& irls_mask =
          *thread_storage->EmptyGridCoverageIrlsMask();
      CHECK_EQ(mask_size, irls_mask.size());

      // Bin features.
      for (const auto& feature : feature_list.feature()) {
        if (feature.irls_weight() > 0) {
          const int x =
              static_cast<int>((feature.x() - shift_x) * scaled_width);
          const int y =
              static_cast<int>((feature.y() - shift_y) * scaled_height);
          // Ignore features that are out of bound in one shifted version.
          if (x < 0 || y < 0 || x >= grid_size || y >= grid_size) {
            continue;
          }

          const int grid_bin = y * grid_size + x;
          irls_mask[grid_bin].push_back(feature.irls_weight());
        }
      }

      for (int k = 0; k < mask_size; ++k) {
        if (irls_mask[k].size() < 2) {  // At least two features present for
          continue;                     // grid cell to be considered.
        }

        const int median_elem = irls_mask[k].size() / 2;
        std::nth_element(irls_mask[k].begin(),
                         irls_mask[k].begin() + median_elem,
                         irls_mask[k].end());

        const float irls_median = irls_mask[k][median_elem];
        const float inlier_score =
            1.0f /
            (1.0f + std::exp(static_cast<double>(
                        -logistic_scale * (irls_median - mid_inlier_score))));
        if (max_features[k] < irls_mask[k].size()) {
          max_features[k] = irls_mask[k].size();
          max_coverage[k] = inlier_score;
        }
      }
    }
  }

  const float cell_weight_sum =
      std::accumulate(grid_cell_weights.begin(), grid_cell_weights.end(), 0.0f);
  CHECK_GT(cell_weight_sum, 0);

  return std::inner_product(max_coverage.begin(), max_coverage.end(),
                            grid_cell_weights.begin(), 0.0f) /
         cell_weight_sum;
}

void MotionEstimation::ComputeMixtureCoverage(
    const RegionFlowFeatureList& feature_list, float min_inlier_score,
    bool assume_rolling_shutter_camera,
    MotionEstimationThreadStorage* thread_storage,
    CameraMotion* camera_motion) const {
  const int grid_size = row_weights_->NumModels();
  const int mask_size = grid_size * grid_size;
  std::vector<float> irls_mask(mask_size, 0.0f);
  std::vector<float> mask_counter(mask_size, 0.0f);

  const float scaled_width = 1.0f / normalized_domain_.x() * (grid_size - 1);
  // Consider features slightly above 1 block distance away from center a block
  // to vote for its coverage.
  const float weight_threshold = row_weights_->WeightThreshold(1.25f);

  const float max_inlier_score = 1.75f * min_inlier_score;
  const float mid_inlier_score = 0.5 * (min_inlier_score + max_inlier_score);

  // Map min_inlier to 0.1 and max_inlier to 0.9 via logistic regression.
  // f(x) = 1 / (1 + exp(-a(x - mid)))
  // f(min) == 0.1 ==> a = ln(1 / 0.1 - 1) / (mid - min)
  const float logistic_scale = 2.1972245 /  // ln(1.0 / 0.1 - 1)
                               (mid_inlier_score - min_inlier_score);

  std::vector<float> texturedness;
  ComputeRegionFlowFeatureTexturedness(feature_list, true, &texturedness);

  int texture_idx = 0;
  for (auto feature = feature_list.feature().begin();
       feature != feature_list.feature().end(); ++feature, ++texture_idx) {
    float irls_weight = feature->irls_weight();
    if (irls_weight == 0) {
      continue;
    }

    // Account for feature texturedness -> low textured outliers do not cause
    // visible artifacts.
    if (assume_rolling_shutter_camera) {
      // Skip low textured outliers.
      if (texturedness[texture_idx] < 0.5 && irls_weight < min_inlier_score) {
        continue;
      }

      // Weight by texture.
      irls_weight /= (texturedness[texture_idx] + 1.e-6f);
    }

    // Interpolate into bins.
    const float x = feature->x() * scaled_width;
    const int bin_x = x;
    const float dx = x - bin_x;
    const int off_x = static_cast<int>(dx != 0);

    const float* row_weights = row_weights_->RowWeights(feature->y());
    for (int k = 0, grid_bin = bin_x; k < row_weights_->NumModels();
         ++k, grid_bin += grid_size) {
      if (row_weights[k] > weight_threshold) {
        irls_mask[grid_bin] += irls_weight * row_weights[k] * (1.0f - dx);
        mask_counter[grid_bin] += row_weights[k] * (1.0f - dx);
        irls_mask[grid_bin + off_x] += irls_weight * row_weights[k] * dx;
        mask_counter[grid_bin + off_x] += row_weights[k] * dx;
      }
    }
  }

  std::vector<float> coverage(grid_size, 0.0f);
  // Record number of occupied cells per block.
  std::vector<int> occupancy(grid_size, 0);

  for (int k = 0, grid_bin = 0; k < grid_size; ++k) {
    for (int l = 0; l < grid_size; ++l, ++grid_bin) {
      // At least two features (at maximum distance from block center)
      // should be present for this cell to be considered.
      if (mask_counter[grid_bin] < 2 * weight_threshold) {
        continue;
      }

      ++occupancy[k];

      const float irls_average = irls_mask[grid_bin] / mask_counter[grid_bin];
      const float inlier_score =
          1.0f /
          (1.0f + std::exp(static_cast<double>(
                      -logistic_scale * (irls_average - mid_inlier_score))));

      coverage[k] += inlier_score;
    }

    // If block was occupied assign small eps coverage so it is not considered
    // empty.
    const float empty_block_eps = 1e-2;
    if (occupancy[k] > 0 && coverage[k] == 0) {
      coverage[k] = empty_block_eps;
    }
  }

  camera_motion->clear_mixture_inlier_coverage();

  // For rolling shutter videos, grid cells without features are assumed to
  // not cause visible distortions (no features -> lack of texture).
  // Limit to 60% of number of cells, to avoid considering only one or two
  // occupied cells stable.
  for (int k = 0; k < grid_size; ++k) {
    const float denom =
        1.0f / (assume_rolling_shutter_camera
                    ? std::max<float>(grid_size * 0.6, occupancy[k])
                    : grid_size);
    camera_motion->add_mixture_inlier_coverage(coverage[k] * denom);
  }
}

bool MotionEstimation::EstimateHomographyIRLS(
    int irls_rounds, bool compute_stability,
    const PriorFeatureWeights* prior_weights,
    MotionEstimationThreadStorage* thread_storage,
    RegionFlowFeatureList* feature_list, CameraMotion* camera_motion) const {
  if (prior_weights && !prior_weights->HasCorrectDimension(
                           irls_rounds, feature_list->feature_size())) {
    LOG(ERROR) << "Prior weights incorrectly initialized, ignoring.";
    prior_weights = nullptr;
  }

  std::unique_ptr<MotionEstimationThreadStorage> local_storage;
  if (thread_storage == nullptr) {
    local_storage.reset(new MotionEstimationThreadStorage(options_, this));
    thread_storage = local_storage.get();
  }

  int num_nonzero_weights =
      feature_list->feature_size() -
      CountIgnoredRegionFlowFeatures(*feature_list, kOutlierIRLSWeight);

  // Use identity if not enough features found.
  const int min_features_for_solution = 9;
  if (num_nonzero_weights < min_features_for_solution) {
    VLOG(1) << "Homography estimation failed, less than "
            << min_features_for_solution << " features usable for estimation.";
    *camera_motion->mutable_homography() = Homography();
    camera_motion->set_flags(camera_motion->flags() |
                             CameraMotion::FLAG_SINGULAR_ESTIMATION);
    return false;
  }

  bool use_float = true;
  // Just declaring does not use memory
  Eigen::Matrix<float, Eigen::Dynamic, 8> matrix_e;
  Eigen::Matrix<float, 8, 1> solution_e;
  Eigen::Matrix<float, 8, 1> rhs_e;
  Eigen::Matrix<double, 8, 8> matrix_d;
  Eigen::Matrix<double, 8, 1> solution_d;
  Eigen::Matrix<double, 8, 1> rhs_d;
  Eigen::Matrix<float, 8, 8> matrix_f;
  Eigen::Matrix<float, 8, 1> solution_f;
  Eigen::Matrix<float, 8, 1> rhs_f;

  if (options_.use_exact_homography_estimation()) {
    const int num_rows =
        2 * feature_list->feature_size() +
        (options_.homography_perspective_regularizer() == 0 ? 0 : 1);
    matrix_e = Eigen::Matrix<float, Eigen::Dynamic, 8>::Zero(num_rows, 8);
    rhs_e = Eigen::Matrix<float, 8, 1>::Zero(8, 1);
    solution_e = Eigen::Matrix<float, 8, 1>::Zero(8, 1);
  } else {
    if (options_.use_highest_accuracy_for_normal_equations()) {
      matrix_d = Eigen::Matrix<double, 8, 8>::Zero(8, 8);
      rhs_d = Eigen::Matrix<double, 8, 1>::Zero();
      solution_d = Eigen::Matrix<double, 8, 1>::Zero(8, 1);
      use_float = false;
    } else {
      matrix_f = Eigen::Matrix<float, 8, 8>::Zero(8, 8);
      rhs_f = Eigen::Matrix<float, 8, 1>::Zero();
      solution_f = Eigen::Matrix<float, 8, 1>::Zero(8, 1);
      use_float = true;
    }
  }

  // Multiple rounds of weighting based L2 optimization.
  Homography norm_model;
  const float irls_residual_scale =
      GetIRLSResidualScale(camera_motion->average_magnitude(),
                           options_.irls_motion_magnitude_fraction());

  const bool irls_use_l0_norm = options_.irls_use_l0_norm();

  const std::vector<float>* irls_priors = nullptr;
  const std::vector<float>* irls_alphas = nullptr;
  if (prior_weights && prior_weights->HasNonZeroAlpha()) {
    irls_priors = &prior_weights->priors;
    irls_alphas = &prior_weights->alphas;
  }

  Homography* prev_solution = nullptr;
  if (options_.homography_exact_denominator_scaling()) {
    prev_solution = &norm_model;
  }

  for (int r = 0; r < irls_rounds; ++r) {
    if (options_.use_exact_homography_estimation()) {
      bool success = false;

      success = HomographyL2QRSolve<float>(
          *feature_list, prev_solution,
          options_.homography_perspective_regularizer(), &matrix_e,
          &solution_e);
      if (!success) {
        VLOG(1) << "Could not solve for homography.";
        *camera_motion->mutable_homography() = Homography();
        camera_motion->set_flags(camera_motion->flags() |
                                 CameraMotion::FLAG_SINGULAR_ESTIMATION);
        return false;
      }
      norm_model =
          HomographyAdapter::FromFloatPointer(solution_e.data(), false);
    } else {
      bool success = false;
      if (options_.use_highest_accuracy_for_normal_equations()) {
        CHECK(!use_float);
        norm_model = HomographyL2NormalEquationSolve<double>(
            *feature_list, prev_solution,
            options_.homography_perspective_regularizer(), &matrix_d, &rhs_d,
            &solution_d, &success);
      } else {
        CHECK(use_float);
        norm_model = HomographyL2NormalEquationSolve<float>(
            *feature_list, prev_solution,
            options_.homography_perspective_regularizer(), &matrix_f, &rhs_f,
            &solution_f, &success);
      }
      if (!success) {
        VLOG(1) << "Could not solve for homography.";
        *camera_motion->mutable_homography() = Homography();
        camera_motion->set_flags(camera_motion->flags() |
                                 CameraMotion::FLAG_SINGULAR_ESTIMATION);
        return false;
      }
    }

    const float alpha = irls_alphas != nullptr ? (*irls_alphas)[r] : 0.0f;
    const float one_minus_alpha = 1.0f - alpha;

    // Compute weights from registration errors.
    const auto feature_start = feature_list->mutable_feature()->begin();
    for (auto feature = feature_start;
         feature != feature_list->mutable_feature()->end(); ++feature) {
      // Ignored features marked as outliers.
      if (feature->irls_weight() == 0.0f) {
        continue;
      }

      // Residual is expressed as geometric difference, that is
      // for a point match (p<->q) with estimated homography p,
      // geometric difference is defined as Hp x q.
      Vector2_f lhs = HomographyAdapter::TransformPoint(
          norm_model, FeatureLocation(*feature));
      // Map to original coordinate system to evaluate error.
      lhs = LinearSimilarityAdapter::TransformPoint(irls_transform_, lhs);
      const Vector3_f lhs3(lhs.x(), lhs.y(), 1);
      const Vector2_f rhs = LinearSimilarityAdapter::TransformPoint(
          irls_transform_, FeatureMatchLocation(*feature));

      const Vector3_f rhs3(rhs.x(), rhs.y(), 1);
      const Vector3_f cross = lhs3.CrossProd(rhs3);
      // We only use the first 2 linearly independent rows.
      const Vector2_f cross2(cross.x(), cross.y());

      const float numerator =
          alpha == 0.0f ? 1.0f
                        : ((*irls_priors)[feature - feature_start] * alpha +
                           one_minus_alpha);

      if (irls_use_l0_norm) {
        feature->set_irls_weight(
            numerator / (cross2.Norm() * irls_residual_scale + kIrlsEps));
      } else {
        feature->set_irls_weight(
            numerator / (std::sqrt(static_cast<double>(cross2.Norm() *
                                                       irls_residual_scale)) +
                         kIrlsEps));
      }
    }
  }

  // Undo pre_transform.
  Homography* model = camera_motion->mutable_homography();
  *model = ModelCompose3(
      LinearSimilarityAdapter::ToHomography(inv_normalization_transform_),
      norm_model,
      LinearSimilarityAdapter::ToHomography(normalization_transform_));

  if (compute_stability) {
    // Score irls and save.
    float average_homography_error = 0;
    // Number of non-zero features.
    int nnz_features = 0;
    const float kMinIrlsWeight = 1e-6f;
    for (const auto& feature : feature_list->feature()) {
      if (feature.irls_weight() > kMinIrlsWeight) {
        if (options_.irls_use_l0_norm()) {
          average_homography_error += 1.0f / feature.irls_weight();
        } else {
          average_homography_error +=
              1.0f / (feature.irls_weight() * feature.irls_weight());
        }
        ++nnz_features;
      }
    }

    if (nnz_features > 0) {
      average_homography_error *= 1.0f / nnz_features;
    }

    camera_motion->set_average_homography_error(average_homography_error);

    // TODO: Use sqrt when use_l0_norm is false.
    // Need to verify that does not break face_compositor before modifying.
    float inlier_threshold =
        options_.stable_homography_bounds().frac_inlier_threshold() *
        hypot(frame_width_, frame_height_);
    camera_motion->set_homography_inlier_coverage(
        GridCoverage(*feature_list, 1.0 / inlier_threshold, thread_storage));
    camera_motion->set_homography_strict_inlier_coverage(GridCoverage(
        *feature_list, options_.strict_coverage_scale() / inlier_threshold,
        thread_storage));
  }
  return true;
}

bool MotionEstimation::MixtureHomographyFromFeature(
    const TranslationModel& camera_translation, int irls_rounds,
    float regularizer, const PriorFeatureWeights* prior_weights,
    RegionFlowFeatureList* feature_list,
    MixtureHomography* mix_homography) const {
  if (prior_weights && !prior_weights->HasCorrectDimension(
                           irls_rounds, feature_list->feature_size())) {
    LOG(ERROR) << "Prior weights incorrectly initialized, ignoring.";
    prior_weights = nullptr;
  }

  const int num_mixtures = options_.num_mixtures();

  // Compute weights if necessary.
  // Compute scale to index mixture weights from normalization.
  CHECK(row_weights_.get() != nullptr);
  CHECK_EQ(row_weights_->YScale(), frame_height_ / normalized_domain_.y());
  CHECK_EQ(row_weights_->NumModels(), num_mixtures);

  const MotionEstimationOptions::MixtureModelMode mixture_mode =
      options_.mixture_model_mode();
  int num_dof = 0;
  int adjacency_constraints = 0;
  switch (mixture_mode) {
    case MotionEstimationOptions::FULL_MIXTURE:
      num_dof = 8 * num_mixtures;
      adjacency_constraints = 8 * (num_mixtures - 1);
      break;
    case MotionEstimationOptions::TRANSLATION_MIXTURE:
      num_dof = 6 + 2 * num_mixtures;
      adjacency_constraints = 2 * (num_mixtures - 1);
      break;
    case MotionEstimationOptions::SKEW_ROTATION_MIXTURE:
      num_dof = 4 + 4 * num_mixtures;
      adjacency_constraints = 4 * (num_mixtures - 1);
      break;
    default:
      LOG(FATAL) << "Unknown MixtureModelMode specified.";
  }

  Eigen::MatrixXf matrix(
      2 * feature_list->feature_size() + adjacency_constraints, num_dof);
  Eigen::MatrixXf solution(num_dof, 1);

  // Multiple rounds of weighting based L2 optimization.
  MixtureHomography norm_model;

  // Initialize with identity.
  for (int k = 0; k < num_mixtures; ++k) {
    norm_model.add_model();
  }

  const bool irls_use_l0_norm = options_.irls_use_l0_norm();

  const std::vector<float>* irls_priors = nullptr;
  const std::vector<float>* irls_alphas = nullptr;
  if (prior_weights && prior_weights->HasNonZeroAlpha()) {
    irls_priors = &prior_weights->priors;
    irls_alphas = &prior_weights->alphas;
  }

  for (int r = 0; r < irls_rounds; ++r) {
    // Unpack solution to mixture homographies, if not full model.
    std::vector<float> solution_unpacked(8 * num_mixtures);
    const float* solution_pointer = &solution_unpacked[0];

    switch (mixture_mode) {
      case MotionEstimationOptions::FULL_MIXTURE:
        if (!MixtureHomographyL2DLTSolve(*feature_list, num_mixtures,
                                         *row_weights_, regularizer, &matrix,
                                         &solution)) {
          return false;
        }
        // No need to unpack solution.
        solution_pointer = solution.data();
        break;

      case MotionEstimationOptions::TRANSLATION_MIXTURE:
        if (!TransMixtureHomographyL2DLTSolve(*feature_list, num_mixtures,
                                              *row_weights_, regularizer,
                                              &matrix, &solution)) {
          return false;
        }
        {
          const float* sol_ptr = solution.data();
          for (int k = 0; k < num_mixtures; ++k) {
            float* curr_ptr = &solution_unpacked[8 * k];
            curr_ptr[0] = sol_ptr[0];
            curr_ptr[1] = sol_ptr[1];
            curr_ptr[2] = sol_ptr[6 + 2 * k];
            curr_ptr[3] = sol_ptr[2];
            curr_ptr[4] = sol_ptr[3];
            curr_ptr[5] = sol_ptr[6 + 2 * k + 1];
            curr_ptr[6] = sol_ptr[4];
            curr_ptr[7] = sol_ptr[5];
          }
        }
        break;

      case MotionEstimationOptions::SKEW_ROTATION_MIXTURE:
        if (!SkewRotMixtureHomographyL2DLTSolve(*feature_list, num_mixtures,
                                                *row_weights_, regularizer,
                                                &matrix, &solution)) {
          return false;
        }
        {
          const float* sol_ptr = solution.data();
          for (int k = 0; k < num_mixtures; ++k) {
            float* curr_ptr = &solution_unpacked[8 * k];
            curr_ptr[0] = sol_ptr[0];
            curr_ptr[1] = sol_ptr[4 + 4 * k];
            curr_ptr[2] = sol_ptr[4 + 4 * k + 2];
            curr_ptr[3] = sol_ptr[4 + 4 * k + 1];
            curr_ptr[4] = sol_ptr[1];
            curr_ptr[5] = sol_ptr[4 + 4 * k + 3];
            curr_ptr[6] = sol_ptr[2];
            curr_ptr[7] = sol_ptr[3];
          }
        }
        break;

      default:
        LOG(FATAL) << "Unknown MixtureModelMode specified.";
    }

    norm_model = MixtureHomographyAdapter::FromFloatPointer(
        solution_pointer, false, 0, num_mixtures);

    const float alpha = irls_alphas != nullptr ? (*irls_alphas)[r] : 0.0f;
    const float one_minus_alpha = 1.0f - alpha;

    // Evaluate IRLS error.
    const auto feature_start = feature_list->mutable_feature()->begin();
    for (auto feature = feature_start;
         feature != feature_list->mutable_feature()->end(); ++feature) {
      if (feature->irls_weight() == 0.0f) {
        continue;
      }

      // Residual is expressed in geometric difference, that is
      // for a point match (p<->q) with estimated homography p,
      // geometric difference is defined as Hp x q.
      Vector2_f lhs = MixtureHomographyAdapter::TransformPoint(
          norm_model, row_weights_->RowWeightsClamped(feature->y()),
          FeatureLocation(*feature));
      // Map to original coordinate system to evaluate error.
      lhs = LinearSimilarityAdapter::TransformPoint(irls_transform_, lhs);

      const Vector3_f lhs3(lhs.x(), lhs.y(), 1);
      const Vector2_f rhs = LinearSimilarityAdapter::TransformPoint(
          irls_transform_, FeatureMatchLocation(*feature));

      const Vector3_f rhs3(rhs.x(), rhs.y(), 1);
      const Vector3_f cross = lhs3.CrossProd(rhs3);

      // We only use the first 2 linearly independent rows.
      const Vector2_f cross2(cross.x(), cross.y());

      const float numerator =
          alpha == 0.0f ? 1.0f
                        : ((*irls_priors)[feature - feature_start] * alpha +
                           one_minus_alpha);

      if (irls_use_l0_norm) {
        feature->set_irls_weight(numerator / (cross2.Norm() + kIrlsEps));
      } else {
        feature->set_irls_weight(
            numerator /
            (std::sqrt(static_cast<double>(cross2.Norm())) + kIrlsEps));
      }
    }
  }

  // Undo pre_transform.
  *mix_homography = MixtureHomographyAdapter::ComposeLeft(
      MixtureHomographyAdapter::ComposeRight(
          norm_model,
          LinearSimilarityAdapter::ToHomography(normalization_transform_)),
      LinearSimilarityAdapter::ToHomography(inv_normalization_transform_));

  switch (mixture_mode) {
    case MotionEstimationOptions::FULL_MIXTURE:
      mix_homography->set_dof(MixtureHomography::ALL_DOF);
      break;
    case MotionEstimationOptions::TRANSLATION_MIXTURE:
      mix_homography->set_dof(MixtureHomography::TRANSLATION_DOF);
      break;
    case MotionEstimationOptions::SKEW_ROTATION_MIXTURE:
      mix_homography->set_dof(MixtureHomography::SKEW_ROTATION_DOF);
      break;
    default:
      LOG(FATAL) << "Unknown MixtureModelMode specified.";
  }
  return true;
}

bool MotionEstimation::EstimateMixtureHomographyIRLS(
    int irls_rounds, bool compute_stability, float regularizer,
    int spectrum_idx, const PriorFeatureWeights* prior_weights,
    MotionEstimationThreadStorage* thread_storage,
    RegionFlowFeatureList* feature_list, CameraMotion* camera_motion) const {
  std::unique_ptr<MotionEstimationThreadStorage> local_storage;
  if (thread_storage == NULL) {
    local_storage.reset(new MotionEstimationThreadStorage(options_, this));
    thread_storage = local_storage.get();
  }

  // We bin features into 3 blocks (top, middle, bottom), requiring each to
  // have sufficient features. This is a specialization of the same test for
  // the homography case. The tested blocks here are not related to the
  // mixture blocks in any manner.
  const int min_features_for_solution = 9;
  const int num_blocks = 3;
  std::vector<int> features_per_block(3, 0);
  const float block_scale = num_blocks / normalized_domain_.y();

  for (const auto& feature : feature_list->feature()) {
    if (feature.irls_weight() > 0) {
      ++features_per_block[feature.y() * block_scale];
    }
  }

  // Require at least two blocks to have sufficient features.
  std::sort(features_per_block.begin(), features_per_block.end());
  if (features_per_block[1] < min_features_for_solution) {
    VLOG(1) << "Mixture homography estimation not possible, less than "
            << min_features_for_solution << " features present.";
    camera_motion->set_flags(camera_motion->flags() |
                             CameraMotion::FLAG_SINGULAR_ESTIMATION);
    return false;
  }

  MixtureHomography mix_homography;
  if (!MixtureHomographyFromFeature(camera_motion->translation(), irls_rounds,
                                    regularizer, prior_weights, feature_list,
                                    &mix_homography)) {
    VLOG(1) << "Non-rigid homography estimated. "
            << "CameraMotion flagged as unstable.";
    camera_motion->set_flags(camera_motion->flags() |
                             CameraMotion::FLAG_SINGULAR_ESTIMATION);
    return false;
  }

  if (compute_stability) {
    // Test if mixture is invertible for every scanline (test via grid,
    // every 10 scanlines, also test one row out of frame domain).
    const float test_grid_size = 10.0f / frame_height_ * normalized_domain_.y();
    bool invertible = true;
    int counter = 0;
    for (float y = -test_grid_size; y < normalized_domain_.y() + test_grid_size;
         y += test_grid_size) {
      ++counter;
      const float* weights = row_weights_->RowWeightsClamped(y);
      Homography test_homography = MixtureHomographyAdapter::ToBaseModel(
          camera_motion->mixture_homography(), weights);
      HomographyAdapter::InvertChecked(test_homography, &invertible);
      if (!invertible) {
        VLOG(1) << "Mixture is not invertible.";
        camera_motion->set_flags(camera_motion->flags() |
                                 CameraMotion::FLAG_SINGULAR_ESTIMATION);
        return false;
      }
    }
  }

  while (spectrum_idx >= camera_motion->mixture_homography_spectrum_size()) {
    camera_motion->add_mixture_homography_spectrum();
  }

  camera_motion->mutable_mixture_homography_spectrum(spectrum_idx)
      ->CopyFrom(mix_homography);

  float mixture_inlier_threshold =
      options_.stable_mixture_homography_bounds().frac_inlier_threshold() *
      hypot(frame_width_, frame_height_);

  // First computed mixture in the spectrum is stored in mixture homography
  // member. Also compute coverage for it.
  if (spectrum_idx == 0) {
    camera_motion->mutable_mixture_homography()->CopyFrom(
        camera_motion->mixture_homography_spectrum(0));
    if (compute_stability) {
      ComputeMixtureCoverage(*feature_list, 1.0f / mixture_inlier_threshold,
                             true, thread_storage, camera_motion);
    }
  }

  // Cap rolling shutter analysis level to be valid level.
  if (options_.mixture_rs_analysis_level() >=
      options_.mixture_regularizer_levels()) {
    LOG(WARNING) << "Resetting mixture_rs_analysis_level to "
                 << options_.mixture_regularizer_levels() - 1;
  }

  const int rs_analysis_level =
      std::min<int>(options_.mixture_rs_analysis_level(),
                    options_.mixture_regularizer_levels() - 1);

  // We compute mixture coverage only for frames which can be safely assumed
  // to be stable mixtures, comparing it to the homography coverage and
  // recording the increase in coverage.
  // Compute coverage assuming rigid camera.
  // TODO: Use sqrt when use_l0_norm is false. Need to verify that
  // does not break face_compositor before modifying.
  if (compute_stability && spectrum_idx == rs_analysis_level) {
    std::vector<float> coverage_backup(
        camera_motion->mixture_inlier_coverage().begin(),
        camera_motion->mixture_inlier_coverage().end());

    // Evaluate mixture coverage and compute rolling shutter guess.
    ComputeMixtureCoverage(*feature_list, 1.0f / mixture_inlier_threshold,
                           false, thread_storage, camera_motion);

    std::vector<float> mixture_inlier_coverage(
        camera_motion->mixture_inlier_coverage().begin(),
        camera_motion->mixture_inlier_coverage().end());

    // Reset to original values.
    if (!coverage_backup.empty()) {
      camera_motion->clear_mixture_inlier_coverage();
      for (float item : coverage_backup) {
        camera_motion->add_mixture_inlier_coverage(item);
      }
    }

    // Estimate rolling shutter score.
    // Use higher threshold on inlier coverage to only consider mixtures that
    // are very reliable.
    const MixtureHomography& rs_mixture =
        camera_motion->mixture_homography_spectrum(
            camera_motion->mixture_homography_spectrum_size() - 1);
    const float rs_stability_threshold =
        options_.stable_mixture_homography_bounds().min_inlier_coverage() *
        1.5f;

    if (IsStableMixtureHomography(rs_mixture, rs_stability_threshold,
                                  mixture_inlier_coverage)) {
      // Only use best matches (strict coverage) to determine by how much
      // mixture models improve on homographies.
      // TODO: Use sqrt when use_l0_norm is false. Need to verify that
      // does not break face_compositor before modifying.
      float homog_inlier_threshold =
          options_.stable_homography_bounds().frac_inlier_threshold() *
          hypot(frame_width_, frame_height_);
      homog_inlier_threshold /= options_.strict_coverage_scale();

      float mixture_coverage = GridCoverage(
          *feature_list, 1.0f / homog_inlier_threshold, thread_storage);

      const float coverage_ratio =
          mixture_coverage /
          (camera_motion->homography_strict_inlier_coverage() + 0.01f);

      camera_motion->set_rolling_shutter_guess(coverage_ratio);
    } else {
      camera_motion->set_rolling_shutter_guess(-1.0f);
    }
  }

  camera_motion->set_mixture_row_sigma(options_.mixture_row_sigma());
  return true;
}

void MotionEstimation::DetermineOverlayIndices(
    bool irls_weights_preinitialized, std::vector<CameraMotion>* camera_motions,
    std::vector<RegionFlowFeatureList*>* feature_lists) const {
  CHECK(camera_motions != nullptr);
  CHECK(feature_lists != nullptr);
  // Two stage estimation: First translation only, followed by
  // overlay analysis.
  const int num_frames = feature_lists->size();
  CHECK_EQ(num_frames, camera_motions->size());

  std::vector<CameraMotion> translation_motions(num_frames);
  const int irls_per_round = options_.irls_rounds();

  // Perform quick initialization of weights and backup original ones.
  if (!irls_weights_preinitialized) {
    for (auto feature_list_ptr : *feature_lists) {
      ResetRegionFlowFeatureIRLSWeights(1.0, feature_list_ptr);
    }
  }

  std::vector<std::vector<float>> original_irls_weights(num_frames);
  for (int f = 0; f < num_frames; ++f) {
    const RegionFlowFeatureList& feature_list = *(*feature_lists)[f];
    GetRegionFlowFeatureIRLSWeights(feature_list, &original_irls_weights[f]);
  }

  ParallelFor(0, num_frames, 1,
              EstimateMotionIRLSInvoker(MODEL_TRANSLATION, irls_per_round,
                                        false, CameraMotion::VALID,
                                        DefaultModelOptions(), this,
                                        nullptr,  // No prior weights.
                                        nullptr,  // No thread storage here.
                                        feature_lists, &translation_motions));

  // Restore weights.
  for (int f = 0; f < num_frames; ++f) {
    RegionFlowFeatureList& feature_list = *(*feature_lists)[f];
    SetRegionFlowFeatureIRLSWeights(original_irls_weights[f], &feature_list);
  }

  const int chunk_size = options_.overlay_analysis_chunk_size();
  const int num_chunks = std::ceil(feature_lists->size() * (1.0f / chunk_size));

  const int overlay_grid_size =
      options_.overlay_detection_options().analysis_mask_size();
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    std::vector<TranslationModel> translations;
    std::vector<RegionFlowFeatureList*> chunk_features;
    const int chunk_start = chunk * chunk_size;
    const int chunk_end = std::min<int>((chunk + 1) * chunk_size, num_frames);
    for (int f = chunk_start; f < chunk_end; ++f) {
      translations.push_back(translation_motions[f].translation());
      chunk_features.push_back((*feature_lists)[f]);
    }

    std::vector<int> overlay_indices;
    OverlayAnalysis(translations, &chunk_features, &overlay_indices);
    for (const auto& overlay_idx : overlay_indices) {
      (*camera_motions)[chunk_start].add_overlay_indices(overlay_idx);
    }

    // Negative marker to frame chunk_start.
    for (int f = chunk_start; f < chunk_end; ++f) {
      if (f > chunk_start) {
        (*camera_motions)[f].add_overlay_indices(chunk_start - f);
      }

      (*camera_motions)[f].set_overlay_domain(overlay_grid_size);
    }
  }
}

// Features are aggregated over a regular grid of the image domain with the
// purpose to determine if a grid bin is deemed part of an overlay or not.
// In particular, we distinguish between two types of overlay features, strict
// and loose ones, based on different thresholds regarding their motion. A grid
// bin is flagged as overlay if it contains a sufficient number of the strict
// features, in which case *all* overlay feature candidates (strict and loose
// ones) are flagged by setting their irls weight to zero.
float MotionEstimation::OverlayAnalysis(
    const std::vector<TranslationModel>& translations,
    std::vector<RegionFlowFeatureList*>* feature_lists,
    std::vector<int>* overlay_indices) const {
  CHECK(feature_lists != nullptr);
  CHECK(overlay_indices != nullptr);
  CHECK_EQ(feature_lists->size(), translations.size());
  overlay_indices->clear();

  const int grid_size =
      options_.overlay_detection_options().analysis_mask_size();

  const int mask_size = grid_size * grid_size;
  const float scaled_width = 1.0f / normalized_domain_.x() * grid_size;
  const float scaled_height = 1.0f / normalized_domain_.y() * grid_size;

  const float strict_zero_motion_threshold =
      options_.overlay_detection_options().strict_near_zero_motion();
  const float strict_max_translation_ratio =
      options_.overlay_detection_options().strict_max_translation_ratio();
  const float loose_zero_motion_threshold =
      options_.overlay_detection_options().loose_near_zero_motion();

  const float strict_min_texturedness =
      options_.overlay_detection_options().strict_min_texturedness();

  std::vector<int> mask_counter(mask_size, 0);
  std::vector<int> overlay_counter(mask_size, 0);
  std::vector<std::vector<RegionFlowFeature*>> overlay_features(mask_size);

  for (int frame = 0; frame < feature_lists->size(); ++frame) {
    const TranslationModel& translation = translations[frame];
    const float trans_magnitude =
        std::hypot(translation.dx(), translation.dy());
    const float strict_trans_threshold =
        strict_max_translation_ratio * trans_magnitude;

    RegionFlowFeatureList* feature_list = (*feature_lists)[frame];
    std::vector<float> texturedness;
    ComputeRegionFlowFeatureTexturedness(*feature_list, false, &texturedness);

    for (int feat_idx = 0, feat_size = feature_list->feature_size();
         feat_idx < feat_size; ++feat_idx) {
      RegionFlowFeature* feature = feature_list->mutable_feature(feat_idx);
      const int x = static_cast<int>(feature->x() * scaled_width);
      const int y = static_cast<int>(feature->y() * scaled_height);
      const int grid_bin = y * grid_size + x;
      ++mask_counter[grid_bin];

      // If translation is near zero, this test is impossible, so continue.
      if (trans_magnitude < 1.0f) {  // In pixels.
        continue;
      }

      const float feat_magnitude =
          LinearSimilarityAdapter::TransformPoint(
              irls_transform_, Vector2_f(feature->dx(), feature->dy()))
              .Norm();
      if (feat_magnitude <= loose_zero_motion_threshold) {
        overlay_features[grid_bin].push_back(feature);
        if (feat_magnitude <= strict_trans_threshold &&
            feat_magnitude <= strict_zero_motion_threshold &&
            texturedness[feat_idx] >= strict_min_texturedness) {
          ++overlay_counter[grid_bin];
        }
      }
    }
  }

  // Determine potential outlier grids.
  const float overlay_min_ratio =
      options_.overlay_detection_options().overlay_min_ratio();

  const float overlay_min_features =
      options_.overlay_detection_options().overlay_min_features();

  for (int i = 0; i < mask_size; ++i) {
    // Ensure sufficient features were aggregated.
    if (mask_counter[i] > overlay_min_features &&
        overlay_counter[i] > overlay_min_ratio * mask_counter[i]) {
      // Consider all features in this bin outliers.
      for (auto& feature_ptr : overlay_features[i]) {
        feature_ptr->set_irls_weight(0.0f);
      }
      overlay_indices->push_back(i);
    }
  }

  return overlay_indices->size() * (1.0f / mask_size);
}

void MotionEstimation::PostIRLSSmoothing(
    const std::vector<CameraMotion>& camera_motions,
    std::vector<RegionFlowFeatureList*>* feature_lists) const {
  CHECK(feature_lists != nullptr);

  std::vector<FeatureGrid<RegionFlowFeature>> feature_grids;
  std::vector<std::vector<int>> feature_taps_3;
  std::vector<std::vector<int>> feature_taps_5;

  struct NonOverlayPredicate {
    bool operator()(const RegionFlowFeature& feature) const {
      return feature.irls_weight() != 0;
    }
  };

  std::vector<RegionFlowFeatureView> feature_views(feature_lists->size());
  for (int k = 0; k < feature_views.size(); ++k) {
    SelectFeaturesFromList(
        [](const RegionFlowFeature& feature) -> bool {
          return feature.irls_weight() != 0;
        },
        (*feature_lists)[k], &feature_views[k]);
  }

  // In normalized domain.
  BuildFeatureGrid(normalized_domain_.x(), normalized_domain_.y(),
                   options_.feature_grid_size(),  // In normalized coords.
                   feature_views, FeatureLocation, &feature_taps_3,
                   &feature_taps_5, nullptr, &feature_grids);

  std::vector<float> feature_frame_confidence(feature_lists->size(), 1.0f);
  if (options_.frame_confidence_weighting()) {
    float max_confidence = 0.0;
    ;
    for (int f = 0; f < feature_lists->size(); ++f) {
      feature_frame_confidence[f] =
          std::max(1e-3f, InlierCoverage(camera_motions[f], false));
      feature_frame_confidence[f] *= feature_frame_confidence[f];
      max_confidence = std::max(max_confidence, feature_frame_confidence[f]);
    }

    const float cut_off_confidence =
        options_.reset_confidence_threshold() * max_confidence;
    for (int f = 0; f < feature_lists->size(); ++f) {
      if (feature_frame_confidence[f] < cut_off_confidence) {
        // If registration is bad, reset to identity and let adjacent
        // frames do the fill in.
        for (auto& feature_ptr : feature_views[f]) {
          feature_ptr->set_irls_weight(1.0f);
        }
      }
    }
  }

  RunTemporalIRLSSmoothing(feature_grids, feature_taps_3, feature_taps_5,
                           feature_frame_confidence, &feature_views);
}

namespace {

void ClearInternalIRLSStructure(RegionFlowFeatureView* feature_view) {
  for (auto& feature_ptr : *feature_view) {
    feature_ptr->clear_internal_irls();
  }
}

}  // namespace.

namespace {

// Note: Push / Pull averaging is performed as reciprocal (effectively we
// average the per feature registration error and convert this back the irls
// weight using 1 / error).
void TemporalIRLSPush(const FeatureGrid<RegionFlowFeature>& curr_grid,
                      const FeatureGrid<RegionFlowFeature>* prev_grid,
                      const std::vector<std::vector<int>>& feature_taps,
                      float space_scale, const std::vector<float>& space_lut,
                      float feature_scale,
                      const std::vector<float>& feature_lut,
                      float temporal_weight, float curr_frame_confidence,
                      float grid_scale, int grid_dim_x,
                      RegionFlowFeatureView* curr_view,
                      RegionFlowFeatureView* prev_view) {
  CHECK(curr_view != nullptr);
  // Spatial filtering of inverse irls weights and the temporally weighted
  // pushed result from the next frame.
  for (auto& feature : *curr_view) {
    float weight_sum = feature->internal_irls().weight_sum() * temporal_weight;
    float value_sum = feature->internal_irls().value_sum() * temporal_weight;

    const int bin_x = feature->x() * grid_scale;
    const int bin_y = feature->y() * grid_scale;
    const int grid_loc = bin_y * grid_dim_x + bin_x;

    for (const auto& bin : feature_taps[grid_loc]) {
      for (const auto& test_feat : curr_grid[bin]) {
        const float dist =
            (FeatureLocation(*test_feat) - FeatureLocation(*feature)).Norm();
        const float feature_dist = RegionFlowFeatureDistance(
            feature->feature_descriptor(), test_feat->feature_descriptor());
        const float weight =
            space_lut[static_cast<int>(dist * space_scale)] *
            feature_lut[static_cast<int>(feature_dist * feature_scale)] *
            curr_frame_confidence;

        weight_sum += weight;
        value_sum += 1.0f / test_feat->irls_weight() * weight;
      }
    }

    // Only zero if spatial AND feature sigma = 0.
    DCHECK_GT(weight_sum, 0);
    feature->mutable_internal_irls()->set_weight_sum(weight_sum);
    feature->mutable_internal_irls()->set_value_sum(value_sum);
  }

  // Clear previous frames interal irls.
  if (prev_view) {
    ClearInternalIRLSStructure(prev_view);
  }

  // Evaluate irls weight and push result to feature in the previous frame along
  // the flow dimension (using spatial interpolation).
  for (auto& feature : *curr_view) {
    feature->set_irls_weight(1.0f / (feature->internal_irls().value_sum() /
                                     feature->internal_irls().weight_sum()));
    feature->clear_internal_irls();

    if (prev_view == NULL) {
      continue;
    }

    const int bin_x = (feature->x() + feature->dx()) * grid_scale;
    const int bin_y = (feature->y() + feature->dy()) * grid_scale;
    const int grid_loc = bin_y * grid_dim_x + bin_x;

    for (const auto& bin : feature_taps[grid_loc]) {
      for (auto& test_feat : (*prev_grid)[bin]) {
        float dist =
            (FeatureLocation(*test_feat) - FeatureMatchLocation(*feature))
                .Norm();
        const float feature_dist =
            RegionFlowFeatureDistance(feature->feature_match_descriptor(),
                                      test_feat->feature_descriptor());
        const float weight =
            space_lut[static_cast<int>(dist * space_scale)] *
            feature_lut[static_cast<int>(feature_dist * feature_scale)];
        TemporalIRLSSmoothing* temporal_irls =
            test_feat->mutable_internal_irls();
        temporal_irls->set_value_sum(temporal_irls->value_sum() +
                                     weight * 1.0f / feature->irls_weight());
        temporal_irls->set_weight_sum(temporal_irls->weight_sum() + weight);
      }
    }
  }
}

void TemporalIRLSPull(const FeatureGrid<RegionFlowFeature>& curr_grid,
                      const FeatureGrid<RegionFlowFeature>& prev_grid,
                      const std::vector<std::vector<int>>& feature_taps,
                      float space_scale, const std::vector<float>& space_lut,
                      float feature_scale,
                      const std::vector<float>& feature_lut,
                      float temporal_weight, float curr_frame_confidence,
                      float grid_scale, int grid_dim_x,
                      RegionFlowFeatureView* curr_view,
                      RegionFlowFeatureView* prev_view) {
  // Pull irls weights of spatially neighboring features from previous frame.
  // Neighborhood is displaced by flow. Pulled weights are weighted by
  // temporal_weight.
  for (auto& feature : *curr_view) {
    const int bin_x = (feature->x() + feature->dx()) * grid_scale;
    const int bin_y = (feature->y() + feature->dy()) * grid_scale;
    const int grid_loc = bin_y * grid_dim_x + bin_x;

    float weight_sum = 0;
    float value_sum = 0;
    for (const auto& bin : feature_taps[grid_loc]) {
      for (const auto& test_feat : prev_grid[bin]) {
        float dist =
            (FeatureLocation(*test_feat) - FeatureMatchLocation(*feature))
                .Norm();
        const float feature_dist =
            RegionFlowFeatureDistance(feature->feature_match_descriptor(),
                                      test_feat->feature_descriptor());
        const float weight =
            space_lut[static_cast<int>(dist * space_scale)] *
            feature_lut[static_cast<int>(feature_dist * feature_scale)];
        weight_sum += weight;
        value_sum += weight * 1.0f / test_feat->irls_weight();
      }
    }

    TemporalIRLSSmoothing* temporal_irls = feature->mutable_internal_irls();
    temporal_irls->set_value_sum(value_sum * temporal_weight);
    temporal_irls->set_weight_sum(weight_sum * temporal_weight);
  }

  // Spatial filtering of neighboring inverse irls_weight and above
  // pulled result from the previous frame.
  for (auto& feature : *curr_view) {
    float weight_sum = feature->internal_irls().weight_sum();
    float value_sum = feature->internal_irls().value_sum();

    const int bin_x = feature->x() * grid_scale;
    const int bin_y = feature->y() * grid_scale;
    const int grid_loc = bin_y * grid_dim_x + bin_x;

    for (const auto& bin : feature_taps[grid_loc]) {
      for (const auto& test_feat : curr_grid[bin]) {
        float dist =
            (FeatureLocation(*test_feat) - FeatureLocation(*feature)).Norm();
        const float feature_dist = RegionFlowFeatureDistance(
            feature->feature_descriptor(), test_feat->feature_descriptor());
        const float weight =
            space_lut[static_cast<int>(dist * space_scale)] *
            feature_lut[static_cast<int>(feature_dist * feature_scale)] *
            curr_frame_confidence;
        weight_sum += weight;
        value_sum += 1.0f / test_feat->irls_weight() * weight;
      }
    }

    CHECK_GT(weight_sum, 0) << feature->irls_weight();
    feature->mutable_internal_irls()->set_weight_sum(weight_sum);
    feature->mutable_internal_irls()->set_value_sum(value_sum);
  }

  // Evaluate irls weight.
  for (auto& feature : *curr_view) {
    feature->set_irls_weight(1.0f / (feature->internal_irls().value_sum() /
                                     feature->internal_irls().weight_sum()));
    feature->clear_internal_irls();
  }
}

}  // namespace.

void MotionEstimation::InitGaussLUT(float sigma, float max_range,
                                    std::vector<float>* lut,
                                    float* scale) const {
  CHECK(lut);
  // Calculate number of bins if scale is non-zero, otherwise use one bin per
  // integer in the domain [0, max_range].
  const int lut_bins = (scale != nullptr) ? (1 << 10) : std::ceil(max_range);
  lut->resize(lut_bins);

  const float bin_size = max_range / lut_bins;
  const float coeff = -0.5f / (sigma * sigma);
  for (int i = 0; i < lut_bins; ++i) {
    const float value = i * bin_size;
    (*lut)[i] = std::exp(value * value * coeff);
  }

  if (scale) {
    *scale = 1.0f / bin_size;
  }
}

// Smooth IRLS weights across the volume.
void MotionEstimation::RunTemporalIRLSSmoothing(
    const std::vector<FeatureGrid<RegionFlowFeature>>& feature_grid,
    const std::vector<std::vector<int>>& feature_taps_3,
    const std::vector<std::vector<int>>& feature_taps_5,
    const std::vector<float>& frame_confidence,
    std::vector<RegionFlowFeatureView>* feature_views) const {
  // Approximate goal for temporal window length with closest integer that
  // achieves roughly homogeneous sized chunks.
  const int temporal_length_goal = options_.temporal_irls_diameter();
  const int num_frames = feature_views->size();

  if (num_frames == 0) {
    return;
  }

  // Clamp IRLS bounds before smoothing, otherwise outliers skew the result
  // heavily.
  for (auto& feature_view : *feature_views) {
    ClampRegionFlowFeatureIRLSWeights(0.01, 100, &feature_view);
  }

  const int num_chunks = std::min<int>(
      1, std::ceil(static_cast<double>(static_cast<float>(num_frames) /
                                       temporal_length_goal)));
  const int temporal_length = std::ceil(
      static_cast<double>(static_cast<float>(num_frames) / num_chunks));

  const float grid_resolution = options_.feature_grid_size();
  const int grid_dim_x =
      std::ceil(static_cast<double>(normalized_domain_.x() / grid_resolution));
  const float grid_scale = 1.0f / grid_resolution;

  const float spatial_sigma = options_.spatial_sigma();

  // Setup Gaussian LUT for smoothing in space, time and feature-space.
  std::vector<float> space_lut;

  // Using 3 tap smoothing, max distance is 2 bin diagonals, for 5 tap
  // smoothing max distance in 3 bin diagonals.
  // We use maximum of 3 * sqrt(2) * bin_radius plus 1% room incase maximum
  // value is attained.
  const float max_space_diff = sqrt(2.0) * 3.f * grid_resolution * 1.01f;
  float space_scale;

  InitGaussLUT(spatial_sigma, max_space_diff, &space_lut, &space_scale);

  const float temporal_sigma = options_.temporal_sigma();
  std::vector<float> temporal_lut;
  InitGaussLUT(temporal_sigma, temporal_length, &temporal_lut, nullptr);

  const float feature_sigma = options_.feature_sigma();
  const float max_feature_diff = sqrt(3.0) * 255.0;  // 3 channels.

  std::vector<float> feature_lut;
  float feature_scale;

  InitGaussLUT(feature_sigma, max_feature_diff, &feature_lut, &feature_scale);

  // Smooth each chunk independently.
  // Smoothing algorithm description:
  //   The volumetric smoothing operation is approximated by a push and pull
  //   phase similar in its nature to scattered data interpolation via push /
  //   pull albeit in time instead of scale space.
  //   (for classic push/pull see:
  //    www.vis.uni-stuttgart.de/~kraus/preprints/vmv06_strengert.pdf)
  //
  //   Our push/pull algorithm pushes and pulls for each feature its irls weight
  //   with gaussian weights (based on proximity in space, time and feature
  //   space) to its neighbors in the previous or next frame. The result of a
  //   push or pull is aggregated in the features TemporalIRLSSmoothing
  //   structure.
  //
  //   In general, we first push the weights through the whole volume (clip)
  //   towards the first frame performing spatial and temporal smoothing, then
  //   pull the resulting weights from the first frame through the whole clip
  //   using again spatial and temporal smoothing.
  //
  //   Specifically, in the push phase a feature's irls weight is updated using
  //   a weigted average (gaussian weights) of its neighboring features and any
  //   pushed information from the next frame
  //   (via TemporalIRLSSmoothing structure).
  //   The updated weight is then pushed along the feature's flow to the
  //   previous frame and spread into the corresponding TemporalIRLSSmoothing
  //   fields of neighboring features in the previous frame.
  //   Similar, the pull phase proceeds by updating a features weights using a
  //   weighted average of its neighboring features and any information from the
  //   previous frame, that is pulled along the features flow.
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int start_frame = chunk * temporal_length;
    const int end_frame = std::min((chunk + 1) * temporal_length, num_frames);

    ClearInternalIRLSStructure(&(*feature_views)[end_frame - 1]);

    // Push pass.
    for (int f = end_frame - 1; f >= start_frame; --f) {
      RegionFlowFeatureView* curr_view = &(*feature_views)[f];
      RegionFlowFeatureView* prev_view =
          f > start_frame ? &(*feature_views)[f - 1] : nullptr;
      const auto& curr_grid = feature_grid[f];
      const auto* prev_grid = f > start_frame ? &feature_grid[f - 1] : nullptr;

      // Evalutate temporal weight (pushed weights to this frames are weighted
      // by temporal weight).
      float temporal_weight = 0;
      for (int e = 1; e < end_frame - f; ++e) {
        temporal_weight += temporal_lut[e];
      }
      // Relative weighting to save multiplication of pushed information,
      // i.e. weight 1.0 for current frame.
      temporal_weight /= temporal_lut[0];

      TemporalIRLSPush(
          curr_grid, prev_grid,
          options_.filter_5_taps() ? feature_taps_5 : feature_taps_3,
          space_scale, space_lut, feature_scale, feature_lut, temporal_weight,
          frame_confidence[f], grid_scale, grid_dim_x, curr_view, prev_view);
    }

    // Pull pass.
    for (int f = start_frame + 1; f < end_frame; ++f) {
      RegionFlowFeatureView* curr_view = &(*feature_views)[f];
      RegionFlowFeatureView* prev_view = &(*feature_views)[f - 1];
      const auto& curr_grid = feature_grid[f];
      const auto& prev_grid = feature_grid[f - 1];

      // Evalutate temporal weight.
      float temporal_weight = 0;
      for (int e = 1; e <= f - start_frame; ++e) {
        temporal_weight += temporal_lut[e];
      }
      // Relative weighting to save multiplication of pushed information.
      temporal_weight /= temporal_lut[0];

      TemporalIRLSPull(
          curr_grid, prev_grid,
          options_.filter_5_taps() ? feature_taps_5 : feature_taps_3,
          space_scale, space_lut, feature_scale, feature_lut, temporal_weight,
          frame_confidence[f], grid_scale, grid_dim_x, curr_view, prev_view);
    }
  }
}

}  // namespace mediapipe
