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

#include "mediapipe/util/tracking/box_detector.h"

#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/opencv_calib3d_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/util/tracking/box_detector.pb.h"
#include "mediapipe/util/tracking/box_tracker.h"
#include "mediapipe/util/tracking/measure_time.h"

namespace mediapipe {

namespace {

void ScaleBox(float scale_x, float scale_y, TimedBoxProto *box) {
  box->set_left(box->left() * scale_x);
  box->set_right(box->right() * scale_x);
  box->set_top(box->top() * scale_y);
  box->set_bottom(box->bottom() * scale_y);

  if (box->has_quad()) {
    for (int c = 0; c < 4; ++c) {
      (*box->mutable_quad()->mutable_vertices())[2 * c] *= scale_x;
      (*box->mutable_quad()->mutable_vertices())[2 * c + 1] *= scale_y;
    }
  }
}

cv::Mat ConvertDescriptorsToMat(const std::vector<std::string> &descriptors) {
  CHECK(!descriptors.empty()) << "empty descriptors.";

  const int descriptors_dims = descriptors[0].size();
  CHECK_GT(descriptors_dims, 0);

  cv::Mat mat(descriptors.size(), descriptors_dims, CV_8U);

  for (int j = 0; j < descriptors.size(); ++j) {
    memcpy(mat.row(j).data, descriptors[j].data(), descriptors_dims);
  }

  return mat;
}

cv::Mat GetDescriptorsWithIndices(const cv::Mat &frame_descriptors,
                                  const std::vector<int> &indices) {
  CHECK_GT(frame_descriptors.rows, 0);

  const int num_inlier_descriptors = indices.size();
  CHECK_GT(num_inlier_descriptors, 0);

  const int descriptors_dims = frame_descriptors.cols;
  CHECK_GT(descriptors_dims, 0);

  cv::Mat mat(num_inlier_descriptors, descriptors_dims, CV_32F);

  for (int j = 0; j < indices.size(); ++j) {
    frame_descriptors.row(indices[j]).copyTo(mat.row(j));
  }

  return mat;
}

}  // namespace

// Using OpenCV brute force matcher along with cross validate match to conduct
// the query.
class BoxDetectorOpencvBfImpl : public BoxDetectorInterface {
 public:
  explicit BoxDetectorOpencvBfImpl(const BoxDetectorOptions &options);

 private:
  std::vector<FeatureCorrespondence> MatchFeatureDescriptors(
      const std::vector<Vector2_f> &features, const cv::Mat &descriptors,
      int box_idx) override;

  cv::BFMatcher bf_matcher_;
};

std::unique_ptr<BoxDetectorInterface> BoxDetectorInterface::Create(
    const BoxDetectorOptions &options) {
  if (options.index_type() == BoxDetectorOptions::OPENCV_BF) {
    return absl::make_unique<BoxDetectorOpencvBfImpl>(options);
  } else {
    LOG(FATAL) << "index type undefined.";
  }
}

BoxDetectorInterface::BoxDetectorInterface(const BoxDetectorOptions &options)
    : options_(options) {}

void BoxDetectorInterface::DetectAndAddBoxFromFeatures(
    const std::vector<Vector2_f> &features, const cv::Mat &descriptors,
    const TimedBoxProtoList &tracked_boxes, int64 timestamp_msec, float scale_x,
    float scale_y, TimedBoxProtoList *detected_boxes) {
  absl::MutexLock lock_access(&access_to_index_);
  image_scale_ = std::min(scale_x, scale_y);
  image_aspect_ = scale_x / scale_y;

  int size_before_add = box_id_to_idx_.size();
  std::vector<bool> tracked(size_before_add, false);
  for (const auto &box : tracked_boxes.box()) {
    if (!box.reacquisition()) {
      continue;
    }

    const absl::flat_hash_map<int, int>::iterator iter =
        box_id_to_idx_.find(box.id());
    if (iter == box_id_to_idx_.end()) {
      // De-normalize the input box to image scale
      TimedBoxProto scaled_box = box;
      ScaleBox(scale_x, scale_y, &scaled_box);

      AddBoxFeaturesToIndex(features, descriptors, scaled_box,
                            /*transform_features_for_pnp*/ true);
    } else {
      int box_idx = iter->second;
      tracked[box_idx] = true;

      float center_x = 0.0f;
      float center_y = 0.0f;
      if (!box.has_quad()) {
        center_x = (box.left() + box.right()) * 0.5f;
        center_y = (box.top() + box.bottom()) * 0.5f;
      } else {
        for (int c = 0; c < 4; ++c) {
          center_x += box.quad().vertices(c * 2);
          center_y += box.quad().vertices(c * 2 + 1);
        }
        center_x /= 4;
        center_y /= 4;
      }

      if (center_x < 0.0f || center_x > 1.0f || center_y < 0.0f ||
          center_y > 1.0f) {
        has_been_out_of_fov_[box_idx] = true;
      }
    }
  }

  for (int idx = 0; idx < size_before_add; ++idx) {
    if ((options_.detect_every_n_frame() > 0 &&
         cnt_detect_called_ % options_.detect_every_n_frame() == 0) ||
        !tracked[idx] ||
        (options_.detect_out_of_fov() && has_been_out_of_fov_[idx])) {
      TimedBoxProtoList det = DetectBox(features, descriptors, idx);
      if (det.box_size() > 0) {
        det.mutable_box(0)->set_time_msec(timestamp_msec);

        // Convert the result box to normalized space.
        ScaleBox(1.0f / scale_x, 1.0f / scale_y, det.mutable_box(0));
        *detected_boxes->add_box() = det.box(0);

        has_been_out_of_fov_[idx] = false;
      }
    }
  }

  // reset timer after detect or add action.
  cnt_detect_called_ = 1;
}

void BoxDetectorInterface::DetectAndAddBox(
    const TrackingData &tracking_data, const TimedBoxProtoList &tracked_boxes,
    int64 timestamp_msec, TimedBoxProtoList *detected_boxes) {
  std::vector<Vector2_f> features_from_tracking_data;
  std::vector<std::string> descriptors_from_tracking_data;
  FeatureAndDescriptorFromTrackingData(tracking_data,
                                       &features_from_tracking_data,
                                       &descriptors_from_tracking_data);

  if (features_from_tracking_data.empty() ||
      descriptors_from_tracking_data.empty()) {
    LOG(WARNING) << "Detection skipped due to empty features or descriptors.";
    return;
  }

  cv::Mat frame_descriptors =
      ConvertDescriptorsToMat(descriptors_from_tracking_data);

  float scale_x, scale_y;
  ScaleFromAspect(tracking_data.frame_aspect(), /*invert*/ false, &scale_x,
                  &scale_y);

  DetectAndAddBoxFromFeatures(features_from_tracking_data, frame_descriptors,
                              tracked_boxes, timestamp_msec, scale_x, scale_y,
                              detected_boxes);
}

bool BoxDetectorInterface::CheckDetectAndAddBox(
    const TimedBoxProtoList &tracked_boxes) {
  bool need_add = false;
  int cnt_tracked = 0;
  for (const auto &box : tracked_boxes.box()) {
    if (!box.reacquisition()) {
      continue;
    }

    const absl::flat_hash_map<int, int>::iterator iter =
        box_id_to_idx_.find(box.id());
    if (iter == box_id_to_idx_.end()) {
      need_add = true;
      break;
    } else {
      ++cnt_tracked;
    }
  }
  // When new boxes being added for reacquisition, we need to run redetection.
  if (need_add) {
    return true;
  }

  const bool is_periodical_check_on = options_.detect_every_n_frame() > 0;
  const bool need_periodical_check =
      is_periodical_check_on &&
      (cnt_detect_called_ % options_.detect_every_n_frame() == 0);

  // When configured to do periodical check, and when need to run the periodical
  // check, we run redetection.
  if (need_periodical_check) {
    return true;
  }

  const bool any_reacquisition_box_missing =
      !box_id_to_idx_.empty() && (cnt_tracked < box_id_to_idx_.size());

  // When NOT configured to use periodical check, we run redetection EVERY frame
  // when any reacquisition box is missing. Note this path of redetection
  // including re-run feature extraction and is expensive, might cause graph
  // throttling on low end devices.
  if (!is_periodical_check_on && any_reacquisition_box_missing) {
    return true;
  }

  // Other cases, increment the cnt_detect_called_ number and return false to
  // not run redetection.
  ++cnt_detect_called_;
  return false;
}

void BoxDetectorInterface::DetectAndAddBox(
    const cv::Mat &image, const TimedBoxProtoList &tracked_boxes,
    int64 timestamp_msec, TimedBoxProtoList *detected_boxes) {
  // Determine if we need execute feature extraction.
  if (!CheckDetectAndAddBox(tracked_boxes)) {
    return;
  }

  const auto &image_query_settings = options_.image_query_settings();

  cv::Mat grayscale;
  if (image.channels() == 3) {
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, grayscale, cv::COLOR_RGBA2GRAY);
  } else {
    grayscale = image;
  }

  cv::Mat resize_image;
  const int longer_edge = std::max(grayscale.cols, grayscale.rows);
  const float longer_edge_scaled = image_query_settings.pyramid_bottom_size();
  if (longer_edge <= longer_edge_scaled) {
    resize_image = grayscale;
  } else {
    float resize_scale = longer_edge_scaled / longer_edge;
    cv::resize(
        grayscale, resize_image,
        cv::Size(resize_scale * grayscale.cols, resize_scale * grayscale.rows),
        cv::INTER_AREA);
  }

  // Use cv::ORB feature extractor for now since it provides better quality of
  // detection results compared with manually constructing pyramid and then use
  // OrbFeatureDescriptor.
  // TODO: Tune OrbFeatureDescriptor to hit similar quality.
  if (!orb_extractor_) {
    orb_extractor_ =
        cv::ORB::create(image_query_settings.max_features(),
                        image_query_settings.pyramid_scale_factor(),
                        image_query_settings.max_pyramid_levels());
  }

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  orb_extractor_->detect(resize_image, keypoints);
  orb_extractor_->compute(resize_image, keypoints, descriptors);

  CHECK_EQ(keypoints.size(), descriptors.rows);

  float inv_scale = 1.0f / std::max(resize_image.cols, resize_image.rows);
  std::vector<Vector2_f> v_keypoints(keypoints.size());
  for (int j = 0; j < keypoints.size(); ++j) {
    v_keypoints[j] =
        Vector2_f(keypoints[j].pt.x * inv_scale, keypoints[j].pt.y * inv_scale);
  }

  float scale_x = resize_image.cols * inv_scale;
  float scale_y = resize_image.rows * inv_scale;

  DetectAndAddBoxFromFeatures(v_keypoints, descriptors, tracked_boxes,
                              timestamp_msec, scale_x, scale_y, detected_boxes);
}

TimedBoxProtoList BoxDetectorInterface::DetectBox(
    const std::vector<Vector2_f> &features, const cv::Mat &descriptors,
    int box_idx) {
  return FindBoxesFromFeatureCorrespondence(
      MatchFeatureDescriptors(features, descriptors, box_idx), box_idx);
}

TimedBoxProtoList BoxDetectorInterface::FindBoxesFromFeatureCorrespondence(
    const std::vector<FeatureCorrespondence> &matches, int box_idx) {
  int max_corr = -1;
  int max_corr_frame = 0;
  for (int j = 0; j < matches.size(); ++j) {
    int num_corr = matches[j].points_frame.size();
    if (num_corr > max_corr) {
      max_corr = num_corr;
      max_corr_frame = j;
    }
  }

  TimedBoxProtoList result_list;

  constexpr int kMinNumCorrespondence = 10;
  if (max_corr < kMinNumCorrespondence) {
    return result_list;
  }

  const TimedBoxProto &ori_box = frame_box_[box_idx][max_corr_frame];
  if (!ori_box.has_quad()) {
    cv::Mat similarity =
        cv::estimateRigidTransform(matches[max_corr_frame].points_index,
                                   matches[max_corr_frame].points_frame, false);

    if (similarity.cols == 0 || similarity.rows == 0) return result_list;

    float similarity_scale =
        std::hypot(similarity.at<double>(0, 0), similarity.at<double>(1, 0));
    float similarity_theta =
        std::atan2(similarity.at<double>(1, 0), similarity.at<double>(0, 0));

    auto *new_box_ptr = result_list.add_box();

    float box_center_x = 0.5f * (ori_box.left() + ori_box.right());
    float box_center_y = 0.5f * (ori_box.top() + ori_box.bottom());

    float new_center_x = similarity.at<double>(0, 0) * box_center_x +
                         similarity.at<double>(0, 1) * box_center_y +
                         similarity.at<double>(0, 2);
    float new_center_y = similarity.at<double>(1, 0) * box_center_x +
                         similarity.at<double>(1, 1) * box_center_y +
                         similarity.at<double>(1, 2);

    new_box_ptr->set_left((ori_box.left() - box_center_x) * similarity_scale +
                          new_center_x);
    new_box_ptr->set_right((ori_box.right() - box_center_x) * similarity_scale +
                           new_center_x);

    new_box_ptr->set_top((ori_box.top() - box_center_y) * similarity_scale +
                         new_center_y);
    new_box_ptr->set_bottom(
        (ori_box.bottom() - box_center_y) * similarity_scale + new_center_y);

    new_box_ptr->set_rotation(ori_box.rotation() + similarity_theta);

    new_box_ptr->set_id(box_idx_to_id_[box_idx]);
    new_box_ptr->set_reacquisition(true);
    return result_list;
  } else {
    return FindQuadFromFeatureCorrespondence(matches[max_corr_frame], ori_box,
                                             image_aspect_);
  }
}

TimedBoxProtoList BoxDetectorInterface::FindQuadFromFeatureCorrespondence(
    const FeatureCorrespondence &matches, const TimedBoxProto &box_proto,
    float frame_aspect) {
  TimedBoxProtoList result_list;

  if (matches.points_frame.size() != matches.points_index.size()) {
    LOG(ERROR) << matches.points_frame.size() << " vs "
               << matches.points_index.size()
               << ". Correpondence size doesn't match.";
    return result_list;
  }

  int matches_size = matches.points_frame.size();
  if (matches_size < options_.min_num_correspondence()) {
    return result_list;
  }

  constexpr int kRansacMaxIterations = 100;
  cv::Mat inliers_set;
  cv::Mat homography = cv::findHomography(
      matches.points_index, matches.points_frame, cv::FM_RANSAC,
      options_.ransac_reprojection_threshold(), inliers_set,
      kRansacMaxIterations);

  // Check if the orientation is preserved, otherwise quad will be flipped.
  double det = homography.at<double>(0, 0) * homography.at<double>(1, 1) -
               homography.at<double>(0, 1) * homography.at<double>(1, 0);
  if (det < 0) {
    return result_list;
  }

  double persp = homography.at<double>(2, 0) * homography.at<double>(2, 0) +
                 homography.at<double>(2, 1) * homography.at<double>(2, 1);
  if (persp > options_.max_perspective_factor()) {
    return result_list;
  }

  std::vector<cv::Point2f> frame_corners;

  if (frame_aspect > 0.0f && box_proto.has_aspect_ratio() &&
      box_proto.aspect_ratio() > 0.0f) {
    float box_scale_x, box_scale_y;
    ScaleFromAspect(box_proto.aspect_ratio(), /*invert*/ false, &box_scale_x,
                    &box_scale_y);
    const float box_half_x = box_scale_x * 0.5;
    const float box_half_y = box_scale_y * 0.5;

    float frame_scale_x, frame_scale_y;
    ScaleFromAspect(frame_aspect, /*invert*/ false, &frame_scale_x,
                    &frame_scale_y);
    const float frame_half_x = frame_scale_x * 0.5;
    const float frame_half_y = frame_scale_y * 0.5;

    std::vector<cv::Point3f> vectors_3d;
    vectors_3d.reserve(matches_size);
    std::vector<cv::Point2f> vectors_2d;
    vectors_2d.reserve(matches_size);

    for (int j = 0; j < matches_size; ++j) {
      if (inliers_set.at<uchar>(j)) {
        vectors_3d.emplace_back(matches.points_index[j].x - box_half_x,
                                matches.points_index[j].y - box_half_y, 0.0f);
        vectors_2d.emplace_back(matches.points_frame[j].x - frame_half_x,
                                matches.points_frame[j].y - frame_half_y);
      }
    }

    constexpr int kMinCorrespondences = 4;
    if (vectors_3d.size() < kMinCorrespondences) {
      return result_list;
    }

    // TODO: Use camera intrinsic if provided.
    cv::Mat rvec, tvec;
    cv::solvePnP(vectors_3d, vectors_2d, cv::Mat::eye(3, 3, CV_64F),
                 cv::Mat::zeros(1, 5, CV_64FC1), rvec, tvec);

    std::vector<cv::Point3f> template_corners{
        cv::Point3f(-box_half_x, -box_half_y, 0),
        cv::Point3f(-box_half_x, box_half_y, 0),
        cv::Point3f(box_half_x, box_half_y, 0),
        cv::Point3f(box_half_x, -box_half_y, 0)};

    cv::projectPoints(template_corners, rvec, tvec, cv::Mat::eye(3, 3, CV_64F),
                      cv::Mat::zeros(1, 5, CV_64FC1), frame_corners);

    for (int j = 0; j < 4; ++j) {
      frame_corners[j].x += frame_half_x;
      frame_corners[j].y += frame_half_y;
    }
  } else {
    std::vector<cv::Point2f> template_corners{
        cv::Point2f(box_proto.quad().vertices(0), box_proto.quad().vertices(1)),
        cv::Point2f(box_proto.quad().vertices(2), box_proto.quad().vertices(3)),
        cv::Point2f(box_proto.quad().vertices(4), box_proto.quad().vertices(5)),
        cv::Point2f(box_proto.quad().vertices(6),
                    box_proto.quad().vertices(7))};

    cv::perspectiveTransform(template_corners, frame_corners, homography);
  }

  auto *new_box_ptr = result_list.add_box();

  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();
  for (int c = 0; c < 4; ++c) {
    new_box_ptr->mutable_quad()->add_vertices(frame_corners[c].x);
    new_box_ptr->mutable_quad()->add_vertices(frame_corners[c].y);

    min_x = std::min(min_x, frame_corners[c].x);
    max_x = std::max(max_x, frame_corners[c].x);
    min_y = std::min(min_y, frame_corners[c].y);
    max_y = std::max(max_y, frame_corners[c].y);
  }

  new_box_ptr->set_left(min_x);
  new_box_ptr->set_right(max_x);
  new_box_ptr->set_top(min_y);
  new_box_ptr->set_bottom(max_y);
  new_box_ptr->set_rotation(0.0f);
  new_box_ptr->set_id(box_proto.id());
  new_box_ptr->set_reacquisition(true);
  if (box_proto.has_aspect_ratio()) {
    new_box_ptr->set_aspect_ratio(box_proto.aspect_ratio());
  }

  return result_list;
}

std::vector<int> BoxDetectorInterface::GetFeatureIndexWithinBox(
    const std::vector<Vector2_f> &features, const TimedBoxProto &box) {
  std::vector<int> insider_idx;
  if (features.empty()) return insider_idx;

  MotionBoxState box_state;
  if (!box.has_quad()) {
    box_state.set_pos_x(box.left());
    box_state.set_pos_y(box.top());
    box_state.set_width(box.right() - box.left());
    box_state.set_height(box.bottom() - box.top());
    box_state.set_rotation(box.rotation());
  } else {
    auto *state_quad_ptr = box_state.mutable_quad();
    for (int c = 0; c < 8; ++c) {
      state_quad_ptr->add_vertices(box.quad().vertices(c));
    }
  }

  const Vector2_f box_scaling(1.0f, 1.0f);
  constexpr float kScaleFactorForBoxEnlarging = 0.1f;
  constexpr int kMinNumFeatures = 60;
  GetFeatureIndicesWithinBox(
      features, box_state, box_scaling,
      /*max_enlarge_size=*/image_scale_ * kScaleFactorForBoxEnlarging,
      /*min_num_features=*/kMinNumFeatures, &insider_idx);
  return insider_idx;
}

void BoxDetectorInterface::AddBoxFeaturesToIndex(
    const std::vector<Vector2_f> &features, const cv::Mat &descriptors,
    const TimedBoxProto &box, bool transform_features_for_pnp) {
  std::vector<int> insider_idx = GetFeatureIndexWithinBox(features, box);

  if (!insider_idx.empty()) {
    const absl::flat_hash_map<int, int>::iterator iter =
        box_id_to_idx_.find(box.id());
    int box_idx;
    if (iter == box_id_to_idx_.end()) {
      box_idx = box_id_to_idx_.size();
      box_id_to_idx_[box.id()] = box_idx;
      box_idx_to_id_.push_back(box.id());
      frame_box_.resize(box_id_to_idx_.size());
      feature_to_frame_.resize(box_id_to_idx_.size());
      feature_keypoints_.resize(box_id_to_idx_.size());
      feature_descriptors_.resize(box_id_to_idx_.size());
      has_been_out_of_fov_.push_back(false);
    } else {
      box_idx = iter->second;
      has_been_out_of_fov_[box_idx] = false;
    }

    // Create a frame
    int frame_id = frame_box_[box_idx].size();
    frame_box_[box_idx].push_back(box);

    cv::Mat box_descriptors =
        GetDescriptorsWithIndices(descriptors, insider_idx);
    if (feature_descriptors_[box_idx].rows == 0) {
      feature_descriptors_[box_idx] = box_descriptors;
    } else {
      cv::vconcat(feature_descriptors_[box_idx], box_descriptors,
                  feature_descriptors_[box_idx]);
    }

    if (box.has_aspect_ratio() && transform_features_for_pnp) {
      // TODO: Dynamically switching between pnp and homography
      // detection is not supported. The detector can only perform detection in
      // one mode in its lifetime.
      float scale_x, scale_y;
      ScaleFromAspect(box.aspect_ratio(), /*invert*/ false, &scale_x, &scale_y);
      std::vector<cv::Point2f> corners_template{
          cv::Point2f(0.0f, 0.0f), cv::Point2f(0.0f, scale_y),
          cv::Point2f(scale_x, scale_y), cv::Point2f(scale_x, 0.0f)};
      std::vector<cv::Point2f> corners_frame(4);
      for (int j = 0; j < 4; ++j) {
        corners_frame[j].x = box.quad().vertices(j * 2);
        corners_frame[j].y = box.quad().vertices(j * 2 + 1);
      }
      cv::Mat h_transform = cv::findHomography(corners_frame, corners_template);
      std::vector<cv::Point2f> features_frame, features_template;
      for (int j = 0; j < insider_idx.size(); ++j) {
        features_frame.emplace_back(features[insider_idx[j]].x(),
                                    features[insider_idx[j]].y());
      }
      cv::perspectiveTransform(features_frame, features_template, h_transform);
      for (int j = 0; j < features_template.size(); ++j) {
        feature_keypoints_[box_idx].emplace_back(features_template[j].x,
                                                 features_template[j].y);
      }
    } else {
      for (int j = 0; j < insider_idx.size(); ++j) {
        feature_keypoints_[box_idx].emplace_back(features[insider_idx[j]]);
      }
    }

    for (int j = 0; j < insider_idx.size(); ++j) {
      feature_to_frame_[box_idx].push_back(frame_id);
    }
  }
}

void BoxDetectorInterface::CancelBoxDetection(int box_id) {
  absl::MutexLock lock_access(&access_to_index_);
  const absl::flat_hash_map<int, int>::iterator iter =
      box_id_to_idx_.find(box_id);
  if (iter == box_id_to_idx_.end()) {
    return;
  } else {
    const int erase_idx = iter->second;
    frame_box_.erase(frame_box_.begin() + erase_idx);
    feature_to_frame_.erase(feature_to_frame_.begin() + erase_idx);
    feature_keypoints_.erase(feature_keypoints_.begin() + erase_idx);
    feature_descriptors_.erase(feature_descriptors_.begin() + erase_idx);
    has_been_out_of_fov_.erase(has_been_out_of_fov_.begin() + erase_idx);
    box_idx_to_id_.erase(box_idx_to_id_.begin() + erase_idx);
    box_id_to_idx_.erase(iter);
    for (int j = erase_idx; j < box_idx_to_id_.size(); ++j) {
      box_id_to_idx_[box_idx_to_id_[j]] = j;
    }
  }
}

BoxDetectorIndex BoxDetectorInterface::ObtainBoxDetectorIndex() const {
  absl::MutexLock lock_access(&access_to_index_);
  BoxDetectorIndex index;
  for (int j = 0; j < frame_box_.size(); ++j) {
    BoxDetectorIndex::BoxEntry *box_ptr = index.add_box_entry();
    for (int i = 0; i < frame_box_[j].size(); ++i) {
      BoxDetectorIndex::BoxEntry::FrameEntry *frame_ptr =
          box_ptr->add_frame_entry();
      *(frame_ptr->mutable_box()) = frame_box_[j][i];
    }

    for (int k = 0; k < feature_to_frame_[j].size(); ++k) {
      BoxDetectorIndex::BoxEntry::FrameEntry *frame_ptr =
          box_ptr->mutable_frame_entry(feature_to_frame_[j][k]);

      frame_ptr->add_keypoints(feature_keypoints_[j][k].x());
      frame_ptr->add_keypoints(feature_keypoints_[j][k].y());
      frame_ptr->add_descriptors()->set_data(
          static_cast<void *>(feature_descriptors_[j].row(k).data),
          feature_descriptors_[j].cols * sizeof(float));
    }
  }

  return index;
}

void BoxDetectorInterface::AddBoxDetectorIndex(const BoxDetectorIndex &index) {
  absl::MutexLock lock_access(&access_to_index_);
  for (int j = 0; j < index.box_entry_size(); ++j) {
    const auto &box_entry = index.box_entry(j);
    for (int i = 0; i < box_entry.frame_entry_size(); ++i) {
      const auto &frame_entry = box_entry.frame_entry(i);

      // If the box to be added already exists in the index, skip.
      if (box_id_to_idx_.find(frame_entry.box().id()) != box_id_to_idx_.end()) {
        continue;
      }

      CHECK_EQ(frame_entry.keypoints_size(),
               frame_entry.descriptors_size() * 2);

      const int num_features = frame_entry.descriptors_size();
      CHECK_GT(num_features, 0);
      std::vector<Vector2_f> features(num_features);

      const int descriptors_dims = frame_entry.descriptors(0).data().size();
      CHECK_GT(descriptors_dims, 0);

      cv::Mat descriptors_mat(num_features, descriptors_dims / sizeof(float),
                              CV_32F);
      for (int k = 0; k < num_features; ++k) {
        features[k] = Vector2_f(frame_entry.keypoints(2 * k),
                                frame_entry.keypoints(2 * k + 1));
        memcpy(descriptors_mat.row(k).data,
               frame_entry.descriptors(k).data().data(), descriptors_dims);
      }

      AddBoxFeaturesToIndex(features, descriptors_mat, frame_entry.box());
    }
  }
}

BoxDetectorOpencvBfImpl::BoxDetectorOpencvBfImpl(
    const BoxDetectorOptions &options)
    : BoxDetectorInterface(options), bf_matcher_(cv::NORM_L2, true) {}

std::vector<FeatureCorrespondence>
BoxDetectorOpencvBfImpl::MatchFeatureDescriptors(
    const std::vector<Vector2_f> &features, const cv::Mat &descriptors,
    int box_idx) {
  CHECK_EQ(features.size(), descriptors.rows);

  std::vector<FeatureCorrespondence> correspondence_result(
      frame_box_[box_idx].size());
  if (features.empty() || descriptors.rows == 0 || descriptors.cols == 0) {
    return correspondence_result;
  }

  int knn = 1;
  std::vector<std::vector<cv::DMatch>> matches;
  bf_matcher_.knnMatch(descriptors, feature_descriptors_[box_idx], matches,
                       knn);

  // Hamming distance threshold for best match distance. This max distance
  // filtering rejects some of false matches which has not been rejected by
  // cross match validation. And the value is determined emprically.
  for (const auto &match_pair : matches) {
    if (match_pair.size() < knn) continue;
    const cv::DMatch &best_match = match_pair[0];
    if (best_match.distance > options_.max_match_distance()) continue;

    int match_idx = feature_to_frame_[box_idx][best_match.trainIdx];

    correspondence_result[match_idx].points_frame.push_back(cv::Point2f(
        features[best_match.queryIdx].x(), features[best_match.queryIdx].y()));
    correspondence_result[match_idx].points_index.push_back(
        cv::Point2f(feature_keypoints_[box_idx][best_match.trainIdx].x(),
                    feature_keypoints_[box_idx][best_match.trainIdx].y()));
  }

  return correspondence_result;
}

}  // namespace mediapipe
