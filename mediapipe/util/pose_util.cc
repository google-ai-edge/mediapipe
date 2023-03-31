#include "mediapipe/util/pose_util.h"

#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace {
// BlazePose 33 landmark names.
enum class PoseLandmarkName {
  kNose = 0,
  kLeftEyeInner,
  kLeftEye,
  kLeftEyeOuter,
  kRightEyeInner,
  kRightEye,
  kRightEyeOuter,
  kLeftEar,
  kRightEar,
  kMouthLeft,
  kMouthRight,
  kLeftShoulder,
  kRightShoulder,
  kLeftElbow,
  kRightElbow,
  kLeftWrist,
  kRightWrist,
  kLeftPinky1,
  kRightPinky1,
  kLeftIndex1,
  kRightIndex1,
  kLeftThumb2,
  kRightThumb2,
  kLeftHip,
  kRightHip,
  kLeftKnee,
  kRightKnee,
  kLeftAnkle,
  kRightAnkle,
  kLeftHeel,
  kRightHeel,
  kLeftFootIndex,
  kRightFootIndex,
};

constexpr int kJointColorMap[33][3] = {
    {0, 0, 255},   {255, 208, 0}, {255, 161, 0}, {255, 114, 0}, {0, 189, 255},
    {0, 236, 255}, {0, 255, 226}, {255, 0, 76},  {0, 255, 131}, {255, 0, 171},
    {0, 255, 37},  {244, 0, 253}, {57, 255, 0},  {151, 0, 255}, {151, 255, 0},
    {57, 0, 255},  {245, 255, 0}, {0, 39, 255},  {255, 169, 0}, {0, 133, 255},
    {255, 75, 0},  {0, 228, 255}, {255, 0, 19},  {0, 255, 189}, {255, 0, 113},
    {0, 255, 94},  {255, 0, 208}, {6, 255, 6},   {207, 0, 255}, {96, 255, 0},
    {112, 0, 255}, {190, 255, 0}, {23, 0, 255}};

constexpr int kJointConnection[35][2] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 7},   {0, 4},   {4, 5},   {5, 6},
    {6, 8},   {9, 10},  {11, 12}, {11, 13}, {13, 15}, {15, 17}, {15, 19},
    {15, 21}, {17, 19}, {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22},
    {18, 20}, {11, 23}, {12, 24}, {23, 24}, {23, 25}, {24, 26}, {25, 27},
    {26, 28}, {27, 29}, {28, 30}, {29, 31}, {30, 32}, {27, 31}, {28, 32}};

const int kConnectionColorMap[35][3] = {
    {127, 104, 127}, {255, 184, 0},   {255, 137, 0},   {255, 57, 38},
    {0, 94, 255},    {0, 212, 255},   {0, 245, 240},   {0, 255, 178},
    {127, 127, 104}, {150, 127, 126}, {197, 0, 254},   {104, 0, 255},
    {28, 19, 255},   {28, 66, 255},   {28, 114, 255},  {0, 86, 255},
    {104, 255, 0},   {198, 255, 0},   {250, 212, 0},   {250, 165, 0},
    {250, 127, 9},   {255, 122, 0},   {122, 127, 221}, {156, 127, 56},
    {127, 127, 151}, {0, 255, 141},   {255, 0, 160},   {3, 255, 50},
    {231, 0, 231},   {51, 255, 3},    {159, 0, 255},   {143, 255, 0},
    {67, 0, 255},    {98, 255, 3},    {115, 0, 255}};
}  // namespace

namespace mediapipe {
void DrawPose(const mediapipe::NormalizedLandmarkList& pose, int target_width,
              int target_height, bool flip_y, cv::Mat* image) {
  constexpr float kVisThres = 0.4f;
  constexpr float kPresThres = 0.4f;
  std::map<int, cv::Point> visible_landmarks;
  for (int j = 0; j < pose.landmark_size(); ++j) {
    const auto& landmark = pose.landmark(j);
    if (landmark.has_visibility() && landmark.visibility() < kVisThres) {
      continue;
    }
    if (landmark.has_presence() && landmark.presence() < kPresThres) {
      continue;
    }
    visible_landmarks[j] = cv::Point(
        landmark.x() * target_width,
        (flip_y ? 1.0f - landmark.y() : landmark.y()) * target_height);
  }

  constexpr int draw_line_width = 5;
  constexpr int draw_circle_radius = 7;
  for (int j = 0; j < 35; ++j) {
    if (visible_landmarks.find(kJointConnection[j][0]) !=
            visible_landmarks.end() &&
        visible_landmarks.find(kJointConnection[j][1]) !=
            visible_landmarks.end()) {
      cv::line(*image, visible_landmarks[kJointConnection[j][0]],
               visible_landmarks[kJointConnection[j][1]],
               cv::Scalar(kConnectionColorMap[j][0], kConnectionColorMap[j][1],
                          kConnectionColorMap[j][2]),
               draw_line_width);
    }
  }

  const int lm = static_cast<int>(PoseLandmarkName::kMouthLeft);
  const int rm = static_cast<int>(PoseLandmarkName::kMouthRight);
  const int ls = static_cast<int>(PoseLandmarkName::kLeftShoulder);
  const int rs = static_cast<int>(PoseLandmarkName::kRightShoulder);
  if (visible_landmarks.find(lm) != visible_landmarks.end() &&
      visible_landmarks.find(rm) != visible_landmarks.end() &&
      visible_landmarks.find(ls) != visible_landmarks.end() &&
      visible_landmarks.find(rs) != visible_landmarks.end()) {
    cv::line(*image, (visible_landmarks[lm] + visible_landmarks[rm]) * 0.5f,
             (visible_landmarks[ls] + visible_landmarks[rs]) * 0.5f,
             cv::Scalar(255, 255, 255), draw_line_width);
  }

  for (const auto& landmark : visible_landmarks) {
    cv::circle(*image, landmark.second, draw_circle_radius,
               cv::Scalar(kJointColorMap[landmark.first][0],
                          kJointColorMap[landmark.first][1],
                          kJointColorMap[landmark.first][2]),
               -1);
  }
}

}  // namespace mediapipe
