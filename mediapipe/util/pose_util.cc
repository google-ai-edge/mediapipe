#include "mediapipe/util/pose_util.h"

#include "absl/log/absl_log.h"
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

const int kFaceMeshLips[40][2] = {
    {61, 146},  {146, 91},  {91, 181},  {181, 84},  {84, 17},   {17, 314},
    {314, 405}, {405, 321}, {321, 375}, {375, 291}, {61, 185},  {185, 40},
    {40, 39},   {39, 37},   {37, 0},    {0, 267},   {267, 269}, {269, 270},
    {270, 409}, {409, 291}, {78, 95},   {95, 88},   {88, 178},  {178, 87},
    {87, 14},   {14, 317},  {317, 402}, {402, 318}, {318, 324}, {324, 308},
    {78, 191},  {191, 80},  {80, 81},   {81, 82},   {82, 13},   {13, 312},
    {312, 311}, {311, 310}, {310, 415}, {415, 308}};

const int kFaceMeshLeftEye[16][2] = {
    {263, 249}, {249, 390}, {390, 373}, {373, 374}, {374, 380}, {380, 381},
    {381, 382}, {382, 362}, {263, 466}, {466, 388}, {388, 387}, {387, 386},
    {386, 385}, {385, 384}, {384, 398}, {398, 362}};

const int kFaceMeshLeftIris[4][2] = {
    {474, 475}, {475, 476}, {476, 477}, {477, 474}};

const int kFaceMeshLeftEyebrow[8][2] = {{276, 283}, {283, 282}, {282, 295},
                                        {295, 285}, {300, 293}, {293, 334},
                                        {334, 296}, {296, 336}};

const int kFaceMeshRightEye[16][2] = {
    {33, 7},    {7, 163},   {163, 144}, {144, 145}, {145, 153}, {153, 154},
    {154, 155}, {155, 133}, {33, 246},  {246, 161}, {161, 160}, {160, 159},
    {159, 158}, {158, 157}, {157, 173}, {173, 133}};

const int kFaceMeshRightEyebrow[8][2] = {{46, 53},  {53, 52}, {52, 65},
                                         {65, 55},  {70, 63}, {63, 105},
                                         {105, 66}, {66, 107}};

const int kFaceMeshRightIris[4][2] = {
    {469, 470}, {470, 471}, {471, 472}, {472, 469}};

const int kFaceMeshFaceOval[36][2] = {
    {10, 338},  {338, 297}, {297, 332}, {332, 284}, {284, 251}, {251, 389},
    {389, 356}, {356, 454}, {454, 323}, {323, 361}, {361, 288}, {288, 397},
    {397, 365}, {365, 379}, {379, 378}, {378, 400}, {400, 377}, {377, 152},
    {152, 148}, {148, 176}, {176, 149}, {149, 150}, {150, 136}, {136, 172},
    {172, 58},  {58, 132},  {132, 93},  {93, 234},  {234, 127}, {127, 162},
    {162, 21},  {21, 54},   {54, 103},  {103, 67},  {67, 109},  {109, 10}};

const int kFaceMeshNose[25][2] = {
    {168, 6},   {6, 197},   {197, 195}, {195, 5},   {5, 4},
    {4, 1},     {1, 19},    {19, 94},   {94, 2},    {98, 97},
    {97, 2},    {2, 326},   {326, 327}, {327, 294}, {294, 278},
    {278, 344}, {344, 440}, {440, 275}, {275, 4},   {4, 45},
    {45, 220},  {220, 115}, {115, 48},  {48, 64},   {64, 98}};

const cv::Scalar kRedColor = cv::Scalar{255, 48, 48};
const cv::Scalar kGreenColor = cv::Scalar{48, 255, 48};
const cv::Scalar kGreenColor2 = cv::Scalar{0, 128, 0};
const cv::Scalar kBlueColor = cv::Scalar{21, 101, 192};
const cv::Scalar kBlueColor2 = cv::Scalar{0, 204, 255};
const cv::Scalar kYellowColor = cv::Scalar{255, 204, 0};
const cv::Scalar kYellowColor2 = cv::Scalar{192, 255, 48};
const cv::Scalar kGrayColor = cv::Scalar{128, 128, 128};
const cv::Scalar kPurpleColor = cv::Scalar{128, 64, 128};
const cv::Scalar kPeachColor = cv::Scalar{255, 229, 180};
const cv::Scalar kWhiteColor = cv::Scalar(224, 224, 224);
const cv::Scalar kCyanColor = cv::Scalar{48, 255, 192};
const cv::Scalar kCyanColor2 = cv::Scalar{48, 48, 255};
const cv::Scalar kMagentaColor = cv::Scalar{255, 48, 192};
const cv::Scalar kPinkColor = cv::Scalar{255, 0, 255};
const cv::Scalar kOrangeColor = cv::Scalar{192, 101, 21};

void ReverseRGB(cv::Scalar* color) {
  int tmp = color->val[0];
  color->val[0] = color->val[2];
  color->val[2] = tmp;
}
}  // namespace

namespace mediapipe {
void DrawPose(const mediapipe::NormalizedLandmarkList& pose, bool flip_y,
              cv::Mat* image) {
  const int target_width = image->cols;
  const int target_height = image->rows;
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

void DrawFace(const mediapipe::NormalizedLandmarkList& face,
              const std::pair<int, int>& image_size, const cv::Mat& affine,
              bool flip_y, bool draw_nose, int color_style, bool reverse_color,
              int draw_line_width, cv::Mat* image) {
  std::vector<cv::Point2f> landmarks;
  for (const auto& lm : face.landmark()) {
    float ori_x = lm.x() * image_size.first;
    float ori_y = (flip_y ? 1.0f - lm.y() : lm.y()) * image_size.second;

    landmarks.emplace_back(
        affine.at<float>(0, 0) * ori_x + affine.at<float>(0, 1) * ori_y +
            affine.at<float>(0, 2),
        affine.at<float>(1, 0) * ori_x + affine.at<float>(1, 1) * ori_y +
            affine.at<float>(1, 2));
  }

  cv::Scalar kFaceOvalColor;
  cv::Scalar kLipsColor;
  cv::Scalar kLeftEyeColor;
  cv::Scalar kLeftEyebrowColor;
  cv::Scalar kLeftEyeIrisColor;
  cv::Scalar kRightEyeColor;
  cv::Scalar kRightEyebrowColor;
  cv::Scalar kRightEyeIrisColor;
  cv::Scalar kNoseColor;
  if (color_style == 0) {
    kFaceOvalColor = kWhiteColor;
    kLipsColor = kWhiteColor;
    kLeftEyeColor = kGreenColor;
    kLeftEyebrowColor = kGreenColor;
    kLeftEyeIrisColor = kGreenColor;
    kRightEyeColor = kRedColor;
    kRightEyebrowColor = kRedColor;
    kRightEyeIrisColor = kRedColor;
    kNoseColor = kWhiteColor;
  } else if (color_style == 1) {
    kFaceOvalColor = kWhiteColor;
    kLipsColor = kBlueColor;
    kLeftEyeColor = kCyanColor;
    kLeftEyebrowColor = kGreenColor;
    kLeftEyeIrisColor = kGreenColor;
    kRightEyeColor = kMagentaColor;
    kRightEyebrowColor = kRedColor;
    kRightEyeIrisColor = kRedColor;
    kNoseColor = kYellowColor;
  } else if (color_style == 2) {
    kFaceOvalColor = kWhiteColor;
    kLipsColor = kRedColor;
    kLeftEyeColor = kYellowColor2;
    kLeftEyebrowColor = kGreenColor;
    kLeftEyeIrisColor = kBlueColor2;
    kRightEyeColor = kPinkColor;
    kRightEyebrowColor = kGreenColor2;
    kRightEyeIrisColor = kCyanColor2;
    kNoseColor = kOrangeColor;
  } else {
    ABSL_LOG(ERROR) << "color_style not supported.";
  }

  if (reverse_color) {
    ReverseRGB(&kFaceOvalColor);
    ReverseRGB(&kLipsColor);
    ReverseRGB(&kLeftEyeColor);
    ReverseRGB(&kLeftEyebrowColor);
    ReverseRGB(&kLeftEyeIrisColor);
    ReverseRGB(&kRightEyeColor);
    ReverseRGB(&kRightEyebrowColor);
    ReverseRGB(&kRightEyeIrisColor);
    ReverseRGB(&kNoseColor);
  }

  for (int j = 0; j < 36; ++j) {
    cv::line(*image, landmarks[kFaceMeshFaceOval[j][0]],
             landmarks[kFaceMeshFaceOval[j][1]], kFaceOvalColor,
             draw_line_width, cv::LINE_AA);
  }

  for (int j = 0; j < 40; ++j) {
    cv::line(*image, landmarks[kFaceMeshLips[j][0]],
             landmarks[kFaceMeshLips[j][1]], kLipsColor, draw_line_width,
             cv::LINE_AA);
  }

  for (int j = 0; j < 16; ++j) {
    cv::line(*image, landmarks[kFaceMeshLeftEye[j][0]],
             landmarks[kFaceMeshLeftEye[j][1]], kLeftEyeColor, draw_line_width,
             cv::LINE_AA);
  }

  for (int j = 0; j < 8; ++j) {
    cv::line(*image, landmarks[kFaceMeshLeftEyebrow[j][0]],
             landmarks[kFaceMeshLeftEyebrow[j][1]], kLeftEyebrowColor,
             draw_line_width, cv::LINE_AA);
  }

  for (int j = 0; j < 4; ++j) {
    cv::line(*image, landmarks[kFaceMeshLeftIris[j][0]],
             landmarks[kFaceMeshLeftIris[j][1]], kLeftEyeIrisColor,
             draw_line_width, cv::LINE_AA);
  }

  for (int j = 0; j < 16; ++j) {
    cv::line(*image, landmarks[kFaceMeshRightEye[j][0]],
             landmarks[kFaceMeshRightEye[j][1]], kRightEyeColor,
             draw_line_width, cv::LINE_AA);
  }

  for (int j = 0; j < 8; ++j) {
    cv::line(*image, landmarks[kFaceMeshRightEyebrow[j][0]],
             landmarks[kFaceMeshRightEyebrow[j][1]], kRightEyebrowColor,
             draw_line_width, cv::LINE_AA);
  }

  for (int j = 0; j < 4; ++j) {
    cv::line(*image, landmarks[kFaceMeshRightIris[j][0]],
             landmarks[kFaceMeshRightIris[j][1]], kRightEyeIrisColor,
             draw_line_width, cv::LINE_AA);
  }

  if (draw_nose) {
    for (int j = 0; j < 25; ++j) {
      cv::line(*image, landmarks[kFaceMeshNose[j][0]],
               landmarks[kFaceMeshNose[j][1]], kNoseColor, draw_line_width,
               cv::LINE_AA);
    }
  }
}
}  // namespace mediapipe
