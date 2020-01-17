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

#include "mediapipe/examples/desktop/autoflip/quality/polynomial_regression_path_solver.h"

#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {
namespace {

// A series of focus point locations from a real video.
constexpr int kNumObservations = 291;
constexpr double data[] = {
    1,   0.4072740648, 2,   0.406096287,  3,   0.4049185093, 4,   0.4037407361,
    5,   0.402562963,  6,   0.4013851806, 7,   0.4002074074, 8,   0.399029625,
    9,   0.3978518519, 10,  0.3966740741, 11,  0.3954963056, 12,  0.3943185231,
    13,  0.39314075,   14,  0.3919629676, 15,  0.3907851944, 16,  0.3896074167,
    17,  0.3884296435, 18,  0.3872518611, 19,  0.386074088,  20,  0.3848963194,
    21,  0.3837185417, 22,  0.3825407685, 23,  0.3813629861, 24,  0.3801852037,
    25,  0.3790074398, 26,  0.3778296574, 27,  0.376651875,  28,  0.3754741111,
    29,  0.3742963287, 30,  0.3731185509, 31,  0.3719407685, 32,  0.3707629861,
    33,  0.3695852222, 34,  0.3684074398, 35,  0.3672296759, 36,  0.3661564352,
    37,  0.3651877315, 38,  0.3643235509, 39,  0.3635639213, 40,  0.3629088241,
    41,  0.36235825,   42,  0.3619122222, 43,  0.3615707269, 44,  0.3613337639,
    45,  0.3612013519, 46,  0.3611734583, 47,  0.3612500926, 48,  0.3614312778,
    49,  0.361717,     50,  0.3621072546, 51,  0.362602037,  52,  0.3632013519,
    53,  0.3639052083, 54,  0.3647136111, 55,  0.3656265417, 56,  0.3666440046,
    57,  0.3677659907, 58,  0.3689925139, 59,  0.3703235926, 60,  0.3716546528,
    61,  0.3729857269, 62,  0.374316787,  63,  0.3756478611, 64,  0.3769789259,
    65,  0.37831,      66,  0.3796410648, 67,  0.3809721296, 68,  0.3823031944,
    69,  0.3836342685, 70,  0.384965338,  71,  0.3862963981, 72,  0.3876274676,
    73,  0.388958537,  74,  0.3902290324, 75,  0.391438963,  76,  0.3925883241,
    77,  0.3936771204, 78,  0.3947053426, 79,  0.3956729954, 80,  0.396580088,
    81,  0.3974265972, 82,  0.3982125509, 83,  0.3989379259, 84,  0.3996027407,
    85,  0.4002069815, 86,  0.400750662,  87,  0.4012337639, 88,  0.4016562963,
    89,  0.4020182731, 90,  0.4023196667, 91,  0.4025604954, 92,  0.4027407593,
    93,  0.4028604491, 94,  0.4029195787, 95,  0.4029181296, 96,  0.4028561204,
    97,  0.4027407593, 98,  0.4026254028, 99,  0.4025100417, 100, 0.4023946852,
    101, 0.4022793241, 102, 0.402163963,  103, 0.4020486065, 104, 0.4019332454,
    105, 0.4018178889, 106, 0.4017025278, 107, 0.4015871759, 108, 0.4014718102,
    109, 0.4013564491, 110, 0.4012410972, 111, 0.4011257315, 112, 0.4010103704,
    113, 0.4008950185, 114, 0.400779662,  115, 0.4008743935, 116, 0.4011792083,
    117, 0.4016941157, 118, 0.4024191111, 119, 0.4033541944, 120, 0.4044993704,
    121, 0.4058546343, 122, 0.4074199815, 123, 0.4091954259, 124, 0.4111809537,
    125, 0.4133765741, 126, 0.415782287,  127, 0.4183980787, 128, 0.4212239676,
    129, 0.4242599352, 130, 0.427506,     131, 0.4309621528, 132, 0.4346283935,
    133, 0.4385047222, 134, 0.4425911407, 135, 0.4468876481, 136, 0.4513942454,
    137, 0.4561109306, 138, 0.4608276204, 139, 0.4655443056, 140, 0.4702609861,
    141, 0.4749776736, 142, 0.4796943574, 143, 0.4844110435, 144, 0.4891277287,
    145, 0.4938444148, 146, 0.4985611,    147, 0.5032777852, 148, 0.5079944708,
    149, 0.512711156,  150, 0.5174278403, 151, 0.5221445259, 152, 0.526861212,
    153, 0.5315778981, 154, 0.5362945833, 155, 0.5410112662, 156, 0.5457279537,
    157, 0.5504446435, 158, 0.5551613241, 159, 0.5598780093, 160, 0.5643503102,
    161, 0.5685782454, 162, 0.572561787,  163, 0.5763009583, 164, 0.5797957546,
    165, 0.583046162,  166, 0.5860521944, 167, 0.5888138611, 168, 0.5913311296,
    169, 0.5936040278, 170, 0.5956325463, 171, 0.5974166806, 172, 0.5989564491,
    173, 0.6002518287, 174, 0.6013028241, 175, 0.6021094491, 176, 0.6026716944,
    177, 0.6029895556, 178, 0.6030630556, 179, 0.602892162,  180, 0.6024768889,
    181, 0.6018172361, 182, 0.6009132083, 183, 0.5997648009, 184, 0.5983720139,
    185, 0.5967348519, 186, 0.595057662,  187, 0.5933804769, 188, 0.5917032963,
    189, 0.5900261065, 190, 0.588348912,  191, 0.5866717315, 192, 0.5849945417,
    193, 0.5833173519, 194, 0.5816401713, 195, 0.5799629861, 196, 0.578285787,
    197, 0.5766086111, 198, 0.5749314213, 199, 0.5732542315, 200, 0.5715770509,
    201, 0.5698998611, 202, 0.5682226667, 203, 0.5665454861, 204, 0.5648682963,
    205, 0.5631911157, 206, 0.5615139213, 207, 0.559989287,  208, 0.5586172083,
    209, 0.5573976713, 210, 0.5563306944, 211, 0.5554162662, 212, 0.5546543889,
    213, 0.5540450648, 214, 0.5535882917, 215, 0.5532840694, 216, 0.5531324028,
    217, 0.5531332824, 218, 0.5532867176, 219, 0.5535927083, 220, 0.5540512454,
    221, 0.5546623333, 222, 0.5554259769, 223, 0.5563421667, 224, 0.5574109148,
    225, 0.5586322083, 226, 0.5600060556, 227, 0.5615324583, 228, 0.563211412,
    229, 0.565042912,  230, 0.5670269722, 231, 0.5691635833, 232, 0.5714527315,
    233, 0.5738944491, 234, 0.576488713,  235, 0.5792355231, 236, 0.581982338,
    237, 0.5847291528, 238, 0.5874759676, 239, 0.5902227824, 240, 0.5929695972,
    241, 0.5957164074, 242, 0.5984632269, 243, 0.601210037,  244, 0.6039568565,
    245, 0.6067036667, 246, 0.6094504815, 247, 0.6121973056, 248, 0.6149441111,
    249, 0.6176909352, 250, 0.62043775,   251, 0.6231845556, 252, 0.6258408148,
    253, 0.628406537,  254, 0.6308817269, 255, 0.6332663426, 256, 0.6355604167,
    257, 0.6377639352, 258, 0.6398769259, 259, 0.6418993519, 260, 0.6438312407,
    261, 0.6456725787, 262, 0.6474233704, 263, 0.6490836111, 264, 0.650653287,
    265, 0.6521324444, 266, 0.653521037,  267, 0.6548190926, 268, 0.656026588,
    269, 0.6571435417, 270, 0.6581699537, 271, 0.6591058102, 272, 0.6599511204,
    273, 0.6607058796, 274, 0.6613700926, 275, 0.6620343056, 276, 0.6626985185,
    277, 0.6633627454, 278, 0.6640269537, 279, 0.6646911759, 280, 0.6653553843,
    281, 0.6660195972, 282, 0.6666838056, 283, 0.6673480278, 284, 0.6680122361,
    285, 0.668676463,  286, 0.6693406713, 287, 0.6700048935, 288, 0.6706691019,
    289, 0.6713333333, 290, 0.671997537,  291, 0.6726617454,
};

constexpr double prediction[] = {
    18.885935,   16.560495,  14.316487,  12.15229,   10.066307,   8.056951,
    6.1226487,   4.2618394,  2.4729967,  0.7545829,  -0.894928,   -2.47702,
    -3.9931893,  -5.4449024, -6.833619,  -8.160782,  -9.427822,   -10.63615,
    -11.78717,   -12.882269, -13.922811, -14.910168, -15.845669,  -16.730648,
    -17.566418,  -18.354284, -19.095533, -19.791435, -20.443249,  -21.052212,
    -21.619564,  -22.146511, -22.634256, -23.08399,  -23.496883,  -23.874086,
    -24.216753,  -24.526012, -24.80297,  -25.048738, -25.264395,  -25.451015,
    -25.609661,  -25.74137,  -25.84718,  -25.928093, -25.985123,  -26.01925,
    -26.031458,  -26.022684, -25.993896, -25.946003, -25.879932,  -25.796581,
    -25.696838,  -25.581581, -25.451654, -25.307919, -25.151188,  -24.982292,
    -24.802023,  -24.611176, -24.410517, -24.200804, -23.98278,   -23.757189,
    -23.52473,   -23.286121, -23.042034, -22.79315,  -22.540123,  -22.283602,
    -22.024214,  -21.762579, -21.499294, -21.234953, -20.970118,  -20.70536,
    -20.441223,  -20.178228, -19.916899, -19.65773,  -19.401217,  -19.147831,
    -18.898027,  -18.65226,  -18.410952, -18.174517, -17.943365,  -17.717875,
    -17.498428,  -17.285383, -17.079079, -16.879856, -16.688019,  -16.503883,
    -16.327726,  -16.15982,  -16.000439, -15.849811, -15.7081785, -15.575754,
    -15.452737,  -15.339321, -15.235674, -15.141964, -15.058327,  -14.9848995,
    -14.9218025, -14.869129, -14.826971, -14.795404, -14.774489,  -14.764267,
    -14.764768,  -14.776015, -14.798016, -14.830744, -14.874178,  -14.9282875,
    -14.993011,  -15.068281, -15.15401,  -15.250105, -15.356457,  -15.472937,
    -15.599405,  -15.73571,  -15.881681, -16.03713,  -16.201872,  -16.37569,
    -16.558355,  -16.749626, -16.94926,  -17.156982, -17.372507,  -17.595535,
    -17.825771,  -18.062872, -18.306505, -18.55632,  -18.811947,  -19.072998,
    -19.339085,  -19.609785, -19.884687, -20.16334,  -20.4453,    -20.730091,
    -21.017235,  -21.306234, -21.59658,  -21.887743, -22.179192,  -22.470362,
    -22.760695,  -23.049604, -23.336494, -23.620754, -23.901754,  -24.17887,
    -24.451435,  -24.718779, -24.980236, -25.235092, -25.482649,  -25.722181,
    -25.952942,  -26.174181, -26.38514,  -26.585024, -26.77304,   -26.948387,
    -27.110231,  -27.257734, -27.39005,  -27.506304, -27.605618,  -27.68709,
    -27.749819,  -27.792877, -27.81533,  -27.816212, -27.794569,  -27.749413,
    -27.679752,  -27.584576, -27.462858, -27.313555, -27.135622,  -26.927996,
    -26.689583,  -26.419294, -26.11602,  -25.778639, -25.40601,   -24.996979,
    -24.550379,  -24.06503,
};

void GenerateDataPointsFromRealVideo(
    const int focus_point_frames_length,
    const int prior_focus_point_frames_length,
    std::vector<FocusPointFrame>* focus_point_frames,
    std::vector<FocusPointFrame>* prior_focus_point_frames) {
  CHECK(focus_point_frames_length + prior_focus_point_frames_length <=
        kNumObservations);
  for (int i = 0; i < prior_focus_point_frames_length; i++) {
    FocusPoint sp;
    sp.set_norm_point_x(data[i]);
    FocusPointFrame spf;
    *spf.add_point() = sp;
    prior_focus_point_frames->push_back(spf);
  }
  for (int i = 0; i < focus_point_frames_length; i++) {
    FocusPoint sp;
    sp.set_norm_point_x(data[i]);
    FocusPointFrame spf;
    *spf.add_point() = sp;
    focus_point_frames->push_back(spf);
  }
}

TEST(PolynomialRegressionPathSolverTest, SuccessInTrackingCameraMode) {
  PolynomialRegressionPathSolver solver;
  std::vector<FocusPointFrame> focus_point_frames;
  std::vector<FocusPointFrame> prior_focus_point_frames;
  std::vector<cv::Mat> all_xforms;
  GenerateDataPointsFromRealVideo(/* focus_point_frames_length = */ 100,
                                  /* prior_focus_point_frames_length = */ 100,
                                  &focus_point_frames,
                                  &prior_focus_point_frames);
  constexpr int kFrameWidth = 200;
  constexpr int kFrameHeight = 300;
  constexpr int kCropWidth = 100;
  constexpr int kCropHeight = 300;
  MP_ASSERT_OK(solver.ComputeCameraPath(
      focus_point_frames, prior_focus_point_frames, kFrameWidth, kFrameHeight,
      kCropWidth, kCropHeight, &all_xforms));
  ASSERT_EQ(all_xforms.size(), 200);
  for (int i = 0; i < all_xforms.size(); i++) {
    cv::Mat mat = all_xforms[i];
    EXPECT_FLOAT_EQ(mat.at<float>(0, 2), prediction[i]);
  }
}

TEST(PolynomialRegressionPathSolverTest, SuccessInStationaryCameraMode) {
  PolynomialRegressionPathSolver solver;
  std::vector<FocusPointFrame> focus_point_frames;
  std::vector<FocusPointFrame> prior_focus_point_frames;
  std::vector<cv::Mat> all_xforms;

  constexpr int kFocusPointFramesLength = 100;
  constexpr float kNormStationaryFocusPointX = 0.34f;

  for (int i = 0; i < kFocusPointFramesLength; i++) {
    FocusPoint sp;
    // Add a fixed normalized focus point location.
    sp.set_norm_point_x(kNormStationaryFocusPointX);
    FocusPointFrame spf;
    *spf.add_point() = sp;
    focus_point_frames.push_back(spf);
  }
  constexpr int kFrameWidth = 300;
  constexpr int kFrameHeight = 300;
  constexpr int kCropWidth = 100;
  constexpr int kCropHeight = 300;
  MP_ASSERT_OK(solver.ComputeCameraPath(
      focus_point_frames, prior_focus_point_frames, kFrameWidth, kFrameHeight,
      kCropWidth, kCropHeight, &all_xforms));
  ASSERT_EQ(all_xforms.size(), kFocusPointFramesLength);

  constexpr int kExpectedShift = -48;
  for (int i = 0; i < all_xforms.size(); i++) {
    cv::Mat mat = all_xforms[i];
    EXPECT_FLOAT_EQ(mat.at<float>(0, 2), kExpectedShift);
  }
}

// Test the case where focus points are so close to boundaries that the amount
// of shifts would have moved the camera to go outside frame boundaries. In this
// case, the solver should regulate the camera position shift to keep it stay
// inside the viewport.
TEST(PolynomialRegressionPathSolverTest, SuccessWhenFocusPointCloseToBoundary) {
  PolynomialRegressionPathSolver solver;
  std::vector<FocusPointFrame> focus_point_frames;
  std::vector<FocusPointFrame> prior_focus_point_frames;
  std::vector<cv::Mat> all_xforms;

  constexpr int kFocusPointFramesLength = 100;
  constexpr float kNormStationaryFocusPointX = 0.99f;

  for (int i = 0; i < kFocusPointFramesLength; i++) {
    FocusPoint sp;
    // Add a fixed normalized focus point location.
    sp.set_norm_point_x(kNormStationaryFocusPointX);
    FocusPointFrame spf;
    *spf.add_point() = sp;
    focus_point_frames.push_back(spf);
  }
  constexpr int kFrameWidth = 500;
  constexpr int kFrameHeight = 300;
  constexpr int kCropWidth = 100;
  constexpr int kCropHeight = 300;
  MP_ASSERT_OK(solver.ComputeCameraPath(
      focus_point_frames, prior_focus_point_frames, kFrameWidth, kFrameHeight,
      kCropWidth, kCropHeight, &all_xforms));
  ASSERT_EQ(all_xforms.size(), kFocusPointFramesLength);

  // Regulate max delta change = (500 - 100) / 2.
  constexpr int kExpectedShift = 200;
  for (int i = 0; i < all_xforms.size(); i++) {
    cv::Mat mat = all_xforms[i];
    EXPECT_FLOAT_EQ(mat.at<float>(0, 2), kExpectedShift);
  }
}

TEST(PolynomialRegressionPathSolverTest, FewFramesShouldWork) {
  PolynomialRegressionPathSolver solver;
  std::vector<FocusPointFrame> focus_point_frames;
  std::vector<FocusPointFrame> prior_focus_point_frames;
  std::vector<cv::Mat> all_xforms;
  GenerateDataPointsFromRealVideo(/* focus_point_frames_length = */ 1,
                                  /* prior_focus_point_frames_length = */ 1,
                                  &focus_point_frames,
                                  &prior_focus_point_frames);
  constexpr int kFrameWidth = 200;
  constexpr int kFrameHeight = 300;
  constexpr int kCropWidth = 100;
  constexpr int kCropHeight = 300;
  MP_ASSERT_OK(solver.ComputeCameraPath(
      focus_point_frames, prior_focus_point_frames, kFrameWidth, kFrameHeight,
      kCropWidth, kCropHeight, &all_xforms));
  ASSERT_EQ(all_xforms.size(), 2);
}

TEST(PolynomialRegressionPathSolverTest, OneCurrentFrameShouldWork) {
  PolynomialRegressionPathSolver solver;
  std::vector<FocusPointFrame> focus_point_frames;
  std::vector<FocusPointFrame> prior_focus_point_frames;
  std::vector<cv::Mat> all_xforms;
  GenerateDataPointsFromRealVideo(/* focus_point_frames_length = */ 1,
                                  /* prior_focus_point_frames_length = */ 0,
                                  &focus_point_frames,
                                  &prior_focus_point_frames);
  constexpr int kFrameWidth = 200;
  constexpr int kFrameHeight = 300;
  constexpr int kCropWidth = 100;
  constexpr int kCropHeight = 300;
  MP_ASSERT_OK(solver.ComputeCameraPath(
      focus_point_frames, prior_focus_point_frames, kFrameWidth, kFrameHeight,
      kCropWidth, kCropHeight, &all_xforms));
  ASSERT_EQ(all_xforms.size(), 1);
}

TEST(PolynomialRegressionPathSolverTest, ZeroFrameShouldFail) {
  PolynomialRegressionPathSolver solver;
  std::vector<FocusPointFrame> focus_point_frames;
  std::vector<FocusPointFrame> prior_focus_point_frames;
  std::vector<cv::Mat> all_xforms;
  GenerateDataPointsFromRealVideo(/* focus_point_frames_length = */ 0,
                                  /* prior_focus_point_frames_length = */ 0,
                                  &focus_point_frames,
                                  &prior_focus_point_frames);
  constexpr int kFrameWidth = 200;
  constexpr int kFrameHeight = 300;
  constexpr int kCropWidth = 100;
  constexpr int kCropHeight = 300;
  ASSERT_FALSE(solver
                   .ComputeCameraPath(focus_point_frames,
                                      prior_focus_point_frames, kFrameWidth,
                                      kFrameHeight, kCropWidth, kCropHeight,
                                      &all_xforms)
                   .ok());
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
