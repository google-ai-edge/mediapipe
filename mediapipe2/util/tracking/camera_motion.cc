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

#include "mediapipe/util/tracking/camera_motion.h"

#include <numeric>

#include "absl/strings/str_format.h"
#include "mediapipe/util/tracking/region_flow.h"

namespace mediapipe {

void CameraMotionToTranslation(const CameraMotion& camera_motion,
                               TranslationModel* model) {
  if (camera_motion.has_translation()) {
    model->CopyFrom(camera_motion.translation());
  } else {
    model->Clear();
  }
}

void CameraMotionToLinearSimilarity(const CameraMotion& camera_motion,
                                    LinearSimilarityModel* model) {
  if (camera_motion.has_linear_similarity()) {
    model->CopyFrom(camera_motion.linear_similarity());
  } else {
    TranslationModel translation;
    CameraMotionToTranslation(camera_motion, &translation);
    model->CopyFrom(LinearSimilarityAdapter::Embed(translation));
  }
}

void CameraMotionToAffine(const CameraMotion& camera_motion,
                          AffineModel* model) {
  if (camera_motion.has_affine()) {
    model->CopyFrom(camera_motion.affine());
  } else {
    LinearSimilarityModel similarity;
    CameraMotionToLinearSimilarity(camera_motion, &similarity);
    model->CopyFrom(AffineAdapter::Embed(similarity));
  }
}

void CameraMotionToHomography(const CameraMotion& camera_motion,
                              Homography* model) {
  if (camera_motion.has_homography()) {
    model->CopyFrom(camera_motion.homography());
  } else {
    AffineModel affine;
    CameraMotionToAffine(camera_motion, &affine);
    model->CopyFrom(HomographyAdapter::Embed(affine));
  }
}

void CameraMotionToMixtureHomography(const CameraMotion& camera_motion,
                                     MixtureHomography* model) {
  if (camera_motion.has_mixture_homography()) {
    model->CopyFrom(camera_motion.mixture_homography());
  } else {
    Homography homography;
    CameraMotionToHomography(camera_motion, &homography);
    model->CopyFrom(MixtureHomographyAdapter::Embed(homography, 1));
  }
}

CameraMotion ComposeCameraMotion(const CameraMotion& lhs,
                                 const CameraMotion& rhs) {
  CHECK_EQ(lhs.frame_width(), rhs.frame_width());
  CHECK_EQ(lhs.frame_height(), rhs.frame_height());

  CameraMotion result = rhs;
  if (lhs.has_translation() || rhs.has_translation()) {
    *result.mutable_translation() =
        ModelCompose2(lhs.translation(), rhs.translation());
  }

  if (lhs.has_similarity() || rhs.has_similarity()) {
    *result.mutable_similarity() =
        ModelCompose2(lhs.similarity(), rhs.similarity());
  }

  if (lhs.has_linear_similarity() || rhs.has_linear_similarity()) {
    *result.mutable_linear_similarity() =
        ModelCompose2(lhs.linear_similarity(), rhs.linear_similarity());
  }

  if (lhs.has_affine() || rhs.has_affine()) {
    *result.mutable_affine() = ModelCompose2(lhs.affine(), rhs.affine());
  }

  if (lhs.has_homography() || rhs.has_homography()) {
    *result.mutable_homography() =
        ModelCompose2(lhs.homography(), rhs.homography());
  }

  if (rhs.has_mixture_homography()) {
    if (lhs.has_mixture_homography()) {
      LOG(ERROR) << "Mixture homographies are not closed under composition, "
                 << "Only rhs mixtures composed with lhs homographies "
                 << "are supported.";
    } else if (lhs.type() <= CameraMotion::UNSTABLE_SIM) {
      // We only composit base model when stability is sufficient.
      *result.mutable_mixture_homography() =
          MixtureHomographyAdapter::ComposeLeft(rhs.mixture_homography(),
                                                lhs.homography());
    }
  } else if (lhs.has_mixture_homography()) {
    LOG(ERROR) << "Only rhs mixtures supported.";
  }

  // Select max unstable type.
  result.set_type(std::max(lhs.type(), rhs.type()));
  result.set_average_magnitude(lhs.average_magnitude() +
                               rhs.average_magnitude());
  result.set_translation_variance(
      std::max(lhs.translation_variance(), rhs.translation_variance()));
  result.set_similarity_inlier_ratio(
      std::min(lhs.similarity_inlier_ratio(), rhs.similarity_inlier_ratio()));

  result.set_similarity_strict_inlier_ratio(
      std::min(lhs.similarity_strict_inlier_ratio(),
               rhs.similarity_strict_inlier_ratio()));

  result.set_average_homography_error(
      std::max(lhs.average_homography_error(), rhs.average_homography_error()));
  result.set_homography_inlier_coverage(std::min(
      lhs.homography_inlier_coverage(), rhs.homography_inlier_coverage()));
  result.set_homography_strict_inlier_coverage(
      std::min(lhs.homography_strict_inlier_coverage(),
               rhs.homography_strict_inlier_coverage()));

  // TODO: Overlay stuff.

  result.set_flags(lhs.flags() | rhs.flags());
  result.set_timestamp_usec(
      std::max(lhs.timestamp_usec(), rhs.timestamp_usec()));
  result.set_match_frame(lhs.match_frame() + rhs.match_frame());

  // TODO: Rest.
  return result;
}

CameraMotion InvertCameraMotion(const CameraMotion& motion) {
  CameraMotion inverted = motion;
  if (motion.has_translation()) {
    *inverted.mutable_translation() = ModelInvert(motion.translation());
  }

  if (motion.has_similarity()) {
    *inverted.mutable_similarity() = ModelInvert(motion.similarity());
  }

  if (motion.has_linear_similarity()) {
    *inverted.mutable_linear_similarity() =
        ModelInvert(motion.linear_similarity());
  }

  if (motion.has_affine()) {
    *inverted.mutable_affine() = ModelInvert(motion.affine());
  }

  if (motion.has_homography()) {
    *inverted.mutable_homography() = ModelInvert(motion.homography());
  }

  if (motion.has_mixture_homography()) {
    LOG(ERROR) << "Mixture homographies are not closed under inversion.";
  }

  return inverted;
}

void SubtractCameraMotionFromFeatures(
    const std::vector<CameraMotion>& camera_motions,
    std::vector<RegionFlowFeatureList*>* feature_lists) {
  CHECK(feature_lists != nullptr);
  CHECK_GE(camera_motions.size(), feature_lists->size());
  if (feature_lists->empty()) {
    return;
  }

  bool use_mixtures = camera_motions[0].has_mixture_homography();

  std::unique_ptr<MixtureRowWeights> row_weights;
  if (use_mixtures) {
    row_weights.reset(MixtureRowWeightsFromCameraMotion(
        camera_motions[0], (*feature_lists)[0]->frame_height()));
  }

  for (int k = 0; k < feature_lists->size(); ++k) {
    Homography background_model;
    MixtureHomography background_model_mixture;
    if (use_mixtures) {
      CameraMotionToMixtureHomography(camera_motions[k],
                                      &background_model_mixture);
    } else {
      CameraMotionToHomography(camera_motions[k], &background_model);
    }

    // Remove motion due to camera motion, leaving only foreground motion.
    for (auto& feature : *(*feature_lists)[k]->mutable_feature()) {
      const Vector2_f background_motion =
          (!use_mixtures ? HomographyAdapter::TransformPoint(
                               background_model, FeatureLocation(feature))
                         : MixtureHomographyAdapter::TransformPoint(
                               background_model_mixture, *row_weights,
                               FeatureLocation(feature))) -
          FeatureLocation(feature);
      const Vector2_f object_motion = FeatureFlow(feature) - background_motion;
      feature.set_dx(object_motion.x());
      feature.set_dy(object_motion.y());
    }
  }
}

float ForegroundMotion(const CameraMotion& camera_motion,
                       const RegionFlowFeatureList& feature_list) {
  if (camera_motion.has_mixture_homography()) {
    LOG(WARNING) << "Mixture homographies are present but function is only "
                 << "using homographies. Truncation error likely.";
  }

  Homography background_motion;
  CameraMotionToHomography(camera_motion, &background_motion);

  float foreground_motion = 0;
  for (auto& feature : feature_list.feature()) {
    const float error = (FeatureMatchLocation(feature) -
                         HomographyAdapter::TransformPoint(
                             background_motion, FeatureLocation(feature)))
                            .Norm();
    foreground_motion += error;
  }

  if (feature_list.feature_size() > 0) {
    foreground_motion *= 1.0f / feature_list.feature_size();
  }

  return foreground_motion;
}

void InitCameraMotionFromFeatureList(const RegionFlowFeatureList& feature_list,
                                     CameraMotion* camera_motion) {
  camera_motion->set_blur_score(feature_list.blur_score());
  camera_motion->set_frac_long_features_rejected(
      feature_list.frac_long_features_rejected());
  camera_motion->set_timestamp_usec(feature_list.timestamp_usec());
  camera_motion->set_match_frame(feature_list.match_frame());
  camera_motion->set_frame_width(feature_list.frame_width());
  camera_motion->set_frame_height(feature_list.frame_height());
}

std::string CameraMotionFlagToString(const CameraMotion& camera_motion) {
  std::string text;
  if (camera_motion.flags() & CameraMotion::FLAG_SHOT_BOUNDARY) {
    text += "SHOT_BOUNDARY|";
  }

  if (camera_motion.flags() & CameraMotion::FLAG_BLURRY_FRAME) {
    text += absl::StrFormat("BLURRY_FRAME %.2f|", camera_motion.bluriness());
  }

  if (camera_motion.flags() & CameraMotion::FLAG_MAJOR_OVERLAY) {
    text += "MAJOR_OVERLAY|";
  }

  if (camera_motion.flags() & CameraMotion::FLAG_SHARP_FRAME) {
    text += "SHARP_FRAME|";
  }

  if (camera_motion.flags() & CameraMotion::FLAG_SHOT_FADE) {
    text += "SHOT_FADE|";
  }

  if (camera_motion.flags() & CameraMotion::FLAG_DUPLICATED) {
    text += "DUPLICATED|";
  }
  return text;
}

std::string CameraMotionTypeToString(const CameraMotion& motion) {
  switch (motion.type()) {
    case CameraMotion::VALID:
      return "VALID";
    case CameraMotion::UNSTABLE_HOMOG:
      return "UNSTABLE_HOMOG";
    case CameraMotion::UNSTABLE_SIM:
      return "UNSTABLE_SIM";
    case CameraMotion::UNSTABLE:
      return "UNSTABLE";
    case CameraMotion::INVALID:
      return "INVALID";
  }

  return "NEVER HAPPENS WITH CLANG";
}

float InlierCoverage(const CameraMotion& camera_motion,
                     bool use_homography_coverage) {
  const int num_block_coverages = camera_motion.mixture_inlier_coverage_size();
  if (num_block_coverages == 0 || use_homography_coverage) {
    if (camera_motion.has_homography()) {
      return camera_motion.homography_inlier_coverage();
    } else {
      return 1.0f;
    }
  } else {
    return std::accumulate(camera_motion.mixture_inlier_coverage().begin(),
                           camera_motion.mixture_inlier_coverage().end(),
                           0.0f) *
           (1.0f / num_block_coverages);
  }
}

template <class CameraMotionContainer>
CameraMotion FirstCameraMotionForLooping(
    const CameraMotionContainer& camera_motions) {
  if (camera_motions.size() < 2) {
    LOG(ERROR) << "Not enough camera motions for refinement.";
    return CameraMotion();
  }

  CameraMotion loop_motion = camera_motions[0];

  // Only update motions present in first camera motion.
  bool has_translation = loop_motion.has_translation();
  bool has_similarity = loop_motion.has_linear_similarity();
  bool has_homography = loop_motion.has_homography();

  TranslationModel translation;
  LinearSimilarityModel similarity;
  Homography homography;

  for (int i = 1; i < camera_motions.size(); ++i) {
    const CameraMotion& motion = camera_motions[i];
    if (motion.has_mixture_homography()) {
      // TODO: Implement
      LOG(WARNING) << "This function does not validly apply mixtures; "
                   << "which are currently not closed under composition. ";
    }

    switch (motion.type()) {
      case CameraMotion::INVALID:
        has_translation = false;
        has_similarity = false;
        has_homography = false;
        break;
      case CameraMotion::UNSTABLE:
        has_similarity = false;
        has_homography = false;
        break;
      case CameraMotion::UNSTABLE_SIM:
        has_homography = false;
        break;
      case CameraMotion::VALID:
      case CameraMotion::UNSTABLE_HOMOG:
        break;
      default:
        LOG(FATAL) << "Unknown CameraMotion::type.";
    }

    // Only accumulate motions which are valid for the entire chain, otherwise
    // keep the pre-initialized motions.
    has_translation =
        has_translation && motion.has_translation() &&
        AccumulateInvertedModel(motion.translation(), &translation);
    has_similarity =
        has_similarity && motion.has_linear_similarity() &&
        AccumulateInvertedModel(motion.linear_similarity(), &similarity);
    has_homography = has_homography && motion.has_homography() &&
                     AccumulateInvertedModel(motion.homography(), &homography);
  }

  if (has_translation) {
    *loop_motion.mutable_translation() = translation;
  }
  if (has_similarity && IsInverseStable(similarity)) {
    *loop_motion.mutable_linear_similarity() = similarity;
  }
  if (has_homography && IsInverseStable(homography)) {
    *loop_motion.mutable_homography() = homography;
  }

  VLOG(1) << "Looping camera motion refinement for "
          << " translation:" << (has_translation ? "successful" : "failed")
          << " similarity:" << (has_similarity ? "successful" : "failed")
          << " homography:" << (has_homography ? "successful" : "failed");

  return loop_motion;
}

// Explicit instantiation.
template CameraMotion FirstCameraMotionForLooping<std::vector<CameraMotion>>(
    const std::vector<CameraMotion>&);
template CameraMotion FirstCameraMotionForLooping<std::deque<CameraMotion>>(
    const std::deque<CameraMotion>&);
}  // namespace mediapipe
