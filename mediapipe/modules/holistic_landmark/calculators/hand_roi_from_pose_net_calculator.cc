// Predicts a hand re-crop ROI directly from body-pose landmarks using a tiny
// learned network (645 params), replacing the hand-coded geometric heuristic.
//
// The network takes six hand-related body-pose keypoints (shoulder, elbow,
// wrist, thumb, index, pinky) and predicts the ROI center, size and rotation in
// one shot. It uses the full (x, y, z) of each keypoint plus engineered
// scale-stable distance features, which lets it estimate a correct ROI even
// when the hand is foreshortened/perpendicular to the camera - the failure mode
// of the original heuristic. Weights are baked into a header, so there is no
// extra model file and inference is a few hundred FLOPs.
//
// See https://github.com/sign-language-processing/mediapipe-hand-crop-fix
#include <cmath>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/holistic_landmark/calculators/hand_roi_from_pose_net_calculator.pb.h"
#include "mediapipe/modules/holistic_landmark/calculators/hand_roi_net_weights.h"

namespace mediapipe {

namespace {
constexpr char kLandmarksTag[] = "LANDMARKS";   // 6: shoulder,elbow,wrist,thumb,index,pinky
constexpr char kImageSizeTag[] = "IMAGE_SIZE";  // std::pair<int,int>
constexpr char kRoiTag[] = "ROI";
// keypoint order within the input landmark list
enum { kShoulder = 0, kElbow, kWrist, kThumb, kIndex, kPinky, kNumKp };
}  // namespace

// Set is_left=true for the left hand: inputs are mirrored to the right-hand
// frame the network was trained on, and the predicted ROI is mirrored back.
class HandRoiFromPoseNetCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Tag(kRoiTag).Set<NormalizedRect>();
    // Optional: set handedness per instantiation via a side packet (lets a
    // shared subgraph reuse this calculator for both hands).
    if (cc->InputSidePackets().HasTag("IS_LEFT")) {
      cc->InputSidePackets().Tag("IS_LEFT").Set<bool>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    is_left_ = cc->Options<HandRoiFromPoseNetCalculatorOptions>().is_left();
    if (cc->InputSidePackets().HasTag("IS_LEFT")) {
      is_left_ = cc->InputSidePackets().Tag("IS_LEFT").Get<bool>();
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kLandmarksTag).IsEmpty()) return absl::OkStatus();
    const auto& lms = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();
    RET_CHECK_GE(lms.landmark_size(), kNumKp);
    const auto& image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    const float W = image_size.first, H = image_size.second;
    const float ar = W / H;

    // Normalized keypoints (mirror x for the left hand to the trained frame).
    float x[kNumKp], y[kNumKp], z[kNumKp];
    for (int i = 0; i < kNumKp; ++i) {
      x[i] = is_left_ ? (1.f - lms.landmark(i).x()) : lms.landmark(i).x();
      y[i] = lms.landmark(i).y();
      z[i] = lms.landmark(i).z();
    }

    // Build the 26-dim feature vector: [ar, (x,y,z)*6, 7 engineered].
    float f[hand_roi_net::kNumFeatures];
    int p = 0;
    f[p++] = ar;
    for (int i = 0; i < kNumKp; ++i) { f[p++] = x[i]; f[p++] = y[i]; f[p++] = z[i]; }
    // Engineered, aspect-corrected (x*ar) distances + z-range (match data.extra_features).
    auto dist = [&](int a, int b) {
      const float dx = (x[a] - x[b]) * ar, dy = y[a] - y[b];
      return std::sqrt(dx * dx + dy * dy);
    };
    f[p++] = dist(kElbow, kWrist);
    f[p++] = dist(kShoulder, kElbow);
    f[p++] = dist(kIndex, kPinky);
    f[p++] = dist(kWrist, kIndex);
    f[p++] = dist(kWrist, kPinky);
    f[p++] = dist(kWrist, kThumb);
    float zmin = z[0], zmax = z[0];
    for (int i = 1; i < kNumKp; ++i) { zmin = std::min(zmin, z[i]); zmax = std::max(zmax, z[i]); }
    f[p++] = zmax - zmin;
    RET_CHECK_EQ(p, hand_roi_net::kNumFeatures);

    float out[hand_roi_net::kNumTargets];
    hand_roi_net::Forward(f, out);  // -> [cx, cy, size, sin, cos], cx/cy absolute

    float cx = out[0], cy = out[1], size = out[2];
    float rotation = std::atan2(out[3], out[4]);  // radians
    if (is_left_) { cx = 1.f - cx; rotation = -rotation; }  // mirror ROI back

    auto roi = absl::make_unique<NormalizedRect>();
    roi->set_x_center(cx);
    roi->set_y_center(cy);
    roi->set_width(size);
    roi->set_height(size * ar);   // square in pixels (size is normalized by width)
    roi->set_rotation(rotation);
    cc->Outputs().Tag(kRoiTag).Add(roi.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  bool is_left_ = false;
};
REGISTER_CALCULATOR(HandRoiFromPoseNetCalculator);

}  // namespace mediapipe
