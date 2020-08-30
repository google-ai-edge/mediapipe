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

#include <cmath>
#include <memory>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

namespace {

constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";
constexpr char kNewEyeLandmarksTag[] = "NEW_EYE_LANDMARKS";
constexpr char kUpdatedFaceLandmarksTag[] = "UPDATED_FACE_LANDMARKS";

constexpr int kNumFaceLandmarks = 468;
// 71 landamrks for left eye and 71 landmarks for right eye.
constexpr int kNumEyeLandmarks = 142;

constexpr int kEyeLandmarkIndicesInFaceLandmarks[] = {
    // Left eye
    // eye lower contour
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    // eye upper contour (excluding corners)
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    // halo x2 lower contour
    130,
    25,
    110,
    24,
    23,
    22,
    26,
    112,
    243,
    // halo x2 upper contour (excluding corners)
    247,
    30,
    29,
    27,
    28,
    56,
    190,
    // halo x3 lower contour
    226,
    31,
    228,
    229,
    230,
    231,
    232,
    233,
    244,
    // halo x3 upper contour (excluding corners)
    113,
    225,
    224,
    223,
    222,
    221,
    189,
    // halo x4 upper contour (no lower because of mesh structure)
    // or eyebrow inner contour
    35,
    124,
    46,
    53,
    52,
    65,
    // halo x5 lower contour
    143,
    111,
    117,
    118,
    119,
    120,
    121,
    128,
    245,
    // halo x5 upper contour (excluding corners)
    // or eyebrow outer contour
    156,
    70,
    63,
    105,
    66,
    107,
    55,
    193,

    // Right eye
    // eye lower contour
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    // eye upper contour (excluding corners)
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    // halo x2 lower contour
    359,
    255,
    339,
    254,
    253,
    252,
    256,
    341,
    463,
    // halo x2 upper contour (excluding corners)
    467,
    260,
    259,
    257,
    258,
    286,
    414,
    // halo x3 lower contour
    446,
    261,
    448,
    449,
    450,
    451,
    452,
    453,
    464,
    // halo x3 upper contour (excluding corners)
    342,
    445,
    444,
    443,
    442,
    441,
    413,
    // halo x4 upper contour (no lower because of mesh structure)
    // or eyebrow inner contour
    265,
    353,
    276,
    283,
    282,
    295,
    // halo x5 lower contour
    372,
    340,
    346,
    347,
    348,
    349,
    350,
    357,
    465,
    // halo x5 upper contour (excluding corners)
    // or eyebrow outer contour
    383,
    300,
    293,
    334,
    296,
    336,
    285,
    417,
};

}  // namespace

// Update face landmarks with new (e.g., refined) values. Currently only updates
// landmarks around the eyes.
//
// Usage example:
// node {
//   calculator: "UpdateFaceLandmarksCalculator"
//   input_stream: "NEW_EYE_LANDMARKS:new_eye_landmarks"
//   input_stream: "FACE_LANDMARKS:face_landmarks"
//   output_stream: "UPDATED_FACE_LANDMARKS:refine_face_landmarks"
// }
//
class UpdateFaceLandmarksCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kFaceLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kNewEyeLandmarksTag).Set<NormalizedLandmarkList>();

    cc->Outputs().Tag(kUpdatedFaceLandmarksTag).Set<NormalizedLandmarkList>();

    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Open(CalculatorContext* cc) {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(UpdateFaceLandmarksCalculator);

::mediapipe::Status UpdateFaceLandmarksCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kFaceLandmarksTag).IsEmpty() ||
      cc->Inputs().Tag(kNewEyeLandmarksTag).IsEmpty()) {
    return ::mediapipe::OkStatus();
  }
  const auto& face_landmarks =
      cc->Inputs().Tag(kFaceLandmarksTag).Get<NormalizedLandmarkList>();
  const auto& new_eye_landmarks =
      cc->Inputs().Tag(kNewEyeLandmarksTag).Get<NormalizedLandmarkList>();

  RET_CHECK_EQ(face_landmarks.landmark_size(), kNumFaceLandmarks)
      << "Wrong number of face landmarks";
  RET_CHECK_EQ(new_eye_landmarks.landmark_size(), kNumEyeLandmarks)
      << "Wrong number of face landmarks";

  auto refined_face_landmarks =
      absl::make_unique<NormalizedLandmarkList>(face_landmarks);
  for (int i = 0; i < kNumEyeLandmarks; ++i) {
    const auto& refined_ld = new_eye_landmarks.landmark(i);
    const int id = kEyeLandmarkIndicesInFaceLandmarks[i];
    refined_face_landmarks->mutable_landmark(id)->set_x(refined_ld.x());
    refined_face_landmarks->mutable_landmark(id)->set_y(refined_ld.y());
    refined_face_landmarks->mutable_landmark(id)->set_z(refined_ld.z());
    refined_face_landmarks->mutable_landmark(id)->set_visibility(
        refined_ld.visibility());
  }
  cc->Outputs()
      .Tag(kUpdatedFaceLandmarksTag)
      .Add(refined_face_landmarks.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
