/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/image_size.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_connections.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_connections.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_roi_refinement_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarks_connections.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/data_renderer.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/googletest.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {
namespace {

using ::mediapipe::api2::builder::GetImageSize;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::core::TaskRunner;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::proto::Approximately;
using ::testing::proto::Partially;

constexpr float kAbsMargin = 0.025;
constexpr absl::string_view kTestDataDirectory =
    "/mediapipe/tasks/testdata/vision/";
constexpr char kHolisticResultFile[] =
    "male_full_height_hands_result_cpu.pbtxt";
constexpr absl::string_view kTestImageFile = "male_full_height_hands.jpg";
constexpr absl::string_view kImageInStream = "image_in";
constexpr absl::string_view kLeftHandLandmarksStream = "left_hand_landmarks";
constexpr absl::string_view kRightHandLandmarksStream = "right_hand_landmarks";
constexpr absl::string_view kFaceLandmarksStream = "face_landmarks";
constexpr absl::string_view kFaceBlendshapesStream = "face_blendshapes";
constexpr absl::string_view kPoseLandmarksStream = "pose_landmarks";
constexpr absl::string_view kRenderedImageOutStream = "rendered_image_out";
constexpr absl::string_view kPoseSegmentationMaskStream =
    "pose_segmentation_mask";
constexpr absl::string_view kHolisticLandmarkerModelBundleFile =
    "holistic_landmarker.task";
constexpr absl::string_view kHandLandmarksModelFile =
    "hand_landmark_full.tflite";
constexpr absl::string_view kHandRoiRefinementModelFile =
    "handrecrop_2020_07_21_v0.f16.tflite";
constexpr absl::string_view kPoseDetectionModelFile = "pose_detection.tflite";
constexpr absl::string_view kPoseLandmarksModelFile =
    "pose_landmark_lite.tflite";
constexpr absl::string_view kFaceDetectionModelFile =
    "face_detection_short_range.tflite";
constexpr absl::string_view kFaceLandmarksModelFile =
    "facemesh2_lite_iris_faceflag_2023_02_14.tflite";
constexpr absl::string_view kFaceBlendshapesModelFile =
    "face_blendshapes.tflite";

enum RenderPart {
  HAND = 0,
  POSE = 1,
  FACE = 2,
};

mediapipe::Color GetColor(RenderPart render_part) {
  mediapipe::Color color;
  switch (render_part) {
    case HAND:
      color.set_b(255);
      color.set_g(255);
      color.set_r(255);
      break;
    case POSE:
      color.set_b(0);
      color.set_g(255);
      color.set_r(0);
      break;
    case FACE:
      color.set_b(0);
      color.set_g(0);
      color.set_r(255);
      break;
  }
  return color;
}

std::string GetFilePath(absl::string_view filename) {
  return file::JoinPath("./", kTestDataDirectory, filename);
}

template <std::size_t N>
mediapipe::LandmarksToRenderDataCalculatorOptions GetRendererOptions(
    const std::array<std::array<int, 2>, N>& connections,
    mediapipe::Color color) {
  mediapipe::LandmarksToRenderDataCalculatorOptions renderer_options;
  for (const auto& connection : connections) {
    renderer_options.add_landmark_connections(connection[0]);
    renderer_options.add_landmark_connections(connection[1]);
  }
  *renderer_options.mutable_landmark_color() = color;
  *renderer_options.mutable_connection_color() = color;
  renderer_options.set_thickness(0.5);
  renderer_options.set_visualize_landmark_depth(false);
  return renderer_options;
}

void ConfigureHandProtoOptions(proto::HolisticLandmarkerGraphOptions& options) {
  options.mutable_hand_landmarks_detector_graph_options()
      ->mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kHandLandmarksModelFile));

  options.mutable_hand_roi_refinement_graph_options()
      ->mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kHandRoiRefinementModelFile));
}

void ConfigureFaceProtoOptions(proto::HolisticLandmarkerGraphOptions& options) {
  // Set face detection model.
  face_detector::proto::FaceDetectorGraphOptions& face_detector_graph_options =
      *options.mutable_face_detector_graph_options();
  face_detector_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kFaceDetectionModelFile));
  face_detector_graph_options.set_num_faces(1);

  // Set face landmarks model.
  face_landmarker::proto::FaceLandmarksDetectorGraphOptions&
      face_landmarks_graph_options =
          *options.mutable_face_landmarks_detector_graph_options();
  face_landmarks_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kFaceLandmarksModelFile));
  face_landmarks_graph_options.mutable_face_blendshapes_graph_options()
      ->mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kFaceBlendshapesModelFile));
}

void ConfigurePoseProtoOptions(proto::HolisticLandmarkerGraphOptions& options) {
  pose_detector::proto::PoseDetectorGraphOptions& pose_detector_graph_options =
      *options.mutable_pose_detector_graph_options();
  pose_detector_graph_options.mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kPoseDetectionModelFile));
  pose_detector_graph_options.set_num_poses(1);
  options.mutable_pose_landmarks_detector_graph_options()
      ->mutable_base_options()
      ->mutable_model_asset()
      ->set_file_name(GetFilePath(kPoseLandmarksModelFile));
}

struct HolisticRequest {
  bool is_left_hand_requested = false;
  bool is_right_hand_requested = false;
  bool is_face_requested = false;
  bool is_face_blendshapes_requested = false;
};

// Helper function to create a TaskRunner.
absl::StatusOr<std::unique_ptr<tasks::core::TaskRunner>> CreateTaskRunner(
    bool use_model_bundle, HolisticRequest holistic_request) {
  Graph graph;

  Stream<Image> image = graph.In("IMAEG").Cast<Image>().SetName(kImageInStream);

  auto& holistic_graph = graph.AddNode(
      "mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph");
  proto::HolisticLandmarkerGraphOptions& options =
      holistic_graph.GetOptions<proto::HolisticLandmarkerGraphOptions>();
  if (use_model_bundle) {
    options.mutable_base_options()->mutable_model_asset()->set_file_name(
        GetFilePath(kHolisticLandmarkerModelBundleFile));
  } else {
    ConfigureHandProtoOptions(options);
    ConfigurePoseProtoOptions(options);
    ConfigureFaceProtoOptions(options);
  }

  std::vector<Stream<mediapipe::RenderData>> render_list;
  image >> holistic_graph.In("IMAGE");
  Stream<std::pair<int, int>> image_size = GetImageSize(image, graph);

  if (holistic_request.is_left_hand_requested) {
    Stream<NormalizedLandmarkList> left_hand_landmarks =
        holistic_graph.Out("LEFT_HAND_LANDMARKS")
            .Cast<NormalizedLandmarkList>()
            .SetName(kLeftHandLandmarksStream);
    Stream<NormalizedRect> left_hand_tracking_roi =
        holistic_graph.Out("LEFT_HAND_TRACKING_ROI").Cast<NormalizedRect>();
    auto left_hand_landmarks_render_data = utils::RenderLandmarks(
        left_hand_landmarks,
        utils::GetRenderScale(image_size, left_hand_tracking_roi, 0.0001,
                              graph),
        GetRendererOptions(hand_landmarker::kHandConnections,
                           GetColor(RenderPart::HAND)),
        graph);
    render_list.push_back(left_hand_landmarks_render_data);
    left_hand_landmarks >> graph.Out("LEFT_HAND_LANDMARKS");
  }
  if (holistic_request.is_right_hand_requested) {
    Stream<NormalizedLandmarkList> right_hand_landmarks =
        holistic_graph.Out("RIGHT_HAND_LANDMARKS")
            .Cast<NormalizedLandmarkList>()
            .SetName(kRightHandLandmarksStream);
    Stream<NormalizedRect> right_hand_tracking_roi =
        holistic_graph.Out("RIGHT_HAND_TRACKING_ROI").Cast<NormalizedRect>();
    auto right_hand_landmarks_render_data = utils::RenderLandmarks(
        right_hand_landmarks,
        utils::GetRenderScale(image_size, right_hand_tracking_roi, 0.0001,
                              graph),
        GetRendererOptions(hand_landmarker::kHandConnections,
                           GetColor(RenderPart::HAND)),
        graph);
    render_list.push_back(right_hand_landmarks_render_data);
    right_hand_landmarks >> graph.Out("RIGHT_HAND_LANDMARKS");
  }
  if (holistic_request.is_face_requested) {
    Stream<NormalizedLandmarkList> face_landmarks =
        holistic_graph.Out("FACE_LANDMARKS")
            .Cast<NormalizedLandmarkList>()
            .SetName(kFaceLandmarksStream);
    Stream<NormalizedRect> face_tracking_roi =
        holistic_graph.Out("FACE_TRACKING_ROI").Cast<NormalizedRect>();
    auto face_landmarks_render_data = utils::RenderLandmarks(
        face_landmarks,
        utils::GetRenderScale(image_size, face_tracking_roi, 0.0001, graph),
        GetRendererOptions(
            face_landmarker::FaceLandmarksConnections::kFaceLandmarksConnectors,
            GetColor(RenderPart::FACE)),
        graph);
    render_list.push_back(face_landmarks_render_data);
    face_landmarks >> graph.Out("FACE_LANDMARKS");
  }
  if (holistic_request.is_face_blendshapes_requested) {
    Stream<ClassificationList> face_blendshapes =
        holistic_graph.Out("FACE_BLENDSHAPES")
            .Cast<ClassificationList>()
            .SetName(kFaceBlendshapesStream);
    face_blendshapes >> graph.Out("FACE_BLENDSHAPES");
  }
  Stream<NormalizedLandmarkList> pose_landmarks =
      holistic_graph.Out("POSE_LANDMARKS")
          .Cast<NormalizedLandmarkList>()
          .SetName(kPoseLandmarksStream);
  Stream<NormalizedRect> pose_tracking_roi =
      holistic_graph.Out("POSE_LANDMARKS_ROI").Cast<NormalizedRect>();
  Stream<Image> pose_segmentation_mask =
      holistic_graph.Out("POSE_SEGMENTATION_MASK")
          .Cast<Image>()
          .SetName(kPoseSegmentationMaskStream);

  auto pose_landmarks_render_data = utils::RenderLandmarks(
      pose_landmarks,
      utils::GetRenderScale(image_size, pose_tracking_roi, 0.0001, graph),
      GetRendererOptions(pose_landmarker::kPoseLandmarksConnections,
                         GetColor(RenderPart::POSE)),
      graph);
  render_list.push_back(pose_landmarks_render_data);
  auto rendered_image =
      utils::Render(
          image, absl::Span<Stream<mediapipe::RenderData>>(render_list), graph)
          .SetName(kRenderedImageOutStream);

  pose_landmarks >> graph.Out("POSE_LANDMARKS");
  pose_segmentation_mask >> graph.Out("POSE_SEGMENTATION_MASK");
  rendered_image >> graph.Out("RENDERED_IMAGE");

  auto config = graph.GetConfig();
  core::FixGraphBackEdges(config);

  return TaskRunner::Create(
      config, std::make_unique<core::MediaPipeBuiltinOpResolver>());
}

template <typename T>
absl::StatusOr<T> FetchResult(const core::PacketMap& output_packets,
                              absl::string_view stream_name) {
  auto it = output_packets.find(std::string(stream_name));
  RET_CHECK(it != output_packets.end());
  return it->second.Get<T>();
}

// Remove fields not to be checked in the result, since the model
// generating expected result is different from the testing model.
void RemoveUncheckedResult(proto::HolisticResult& holistic_result) {
  for (auto& landmark :
       *holistic_result.mutable_pose_landmarks()->mutable_landmark()) {
    landmark.clear_z();
    landmark.clear_visibility();
    landmark.clear_presence();
  }
  for (auto& landmark :
       *holistic_result.mutable_face_landmarks()->mutable_landmark()) {
    landmark.clear_z();
    landmark.clear_visibility();
    landmark.clear_presence();
  }
  for (auto& landmark :
       *holistic_result.mutable_left_hand_landmarks()->mutable_landmark()) {
    landmark.clear_z();
    landmark.clear_visibility();
    landmark.clear_presence();
  }
  for (auto& landmark :
       *holistic_result.mutable_right_hand_landmarks()->mutable_landmark()) {
    landmark.clear_z();
    landmark.clear_visibility();
    landmark.clear_presence();
  }
}

std::string RequestToString(HolisticRequest request) {
  return absl::StrFormat(
      "%s_%s_%s_%s",
      request.is_left_hand_requested ? "left_hand" : "no_left_hand",
      request.is_right_hand_requested ? "right_hand" : "no_right_hand",
      request.is_face_requested ? "face" : "no_face",
      request.is_face_blendshapes_requested ? "face_blendshapes"
                                            : "no_face_blendshapes");
}

struct TestParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of test image.
  std::string test_image_name;
  // Whether to use holistic model bundle to test.
  bool use_model_bundle;
  // Requests of holistic parts.
  HolisticRequest holistic_request;
};

class SmokeTest : public testing::TestWithParam<TestParams> {};

TEST_P(SmokeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(GetFilePath(GetParam().test_image_name)));

  proto::HolisticResult holistic_result;
  MP_ASSERT_OK(GetTextProto(GetFilePath(kHolisticResultFile), &holistic_result,
                            ::file::Defaults()));
  RemoveUncheckedResult(holistic_result);

  MP_ASSERT_OK_AND_ASSIGN(auto task_runner,
                          CreateTaskRunner(GetParam().use_model_bundle,
                                           GetParam().holistic_request));
  MP_ASSERT_OK_AND_ASSIGN(auto output_packets,
                          task_runner->Process({{std::string(kImageInStream),
                                                 MakePacket<Image>(image)}}));

  // Check face landmarks
  if (GetParam().holistic_request.is_face_requested) {
    MP_ASSERT_OK_AND_ASSIGN(auto face_landmarks,
                            FetchResult<NormalizedLandmarkList>(
                                output_packets, kFaceLandmarksStream));
    EXPECT_THAT(
        face_landmarks,
        Approximately(Partially(EqualsProto(holistic_result.face_landmarks())),
                      /*margin=*/kAbsMargin));
  } else {
    ASSERT_FALSE(output_packets.contains(std::string(kFaceLandmarksStream)));
  }

  if (GetParam().holistic_request.is_face_blendshapes_requested) {
    MP_ASSERT_OK_AND_ASSIGN(auto face_blendshapes,
                            FetchResult<ClassificationList>(
                                output_packets, kFaceBlendshapesStream));
    EXPECT_THAT(face_blendshapes,
                Approximately(
                    Partially(EqualsProto(holistic_result.face_blendshapes())),
                    /*margin=*/kAbsMargin));
  } else {
    ASSERT_FALSE(output_packets.contains(std::string(kFaceBlendshapesStream)));
  }

  // Check Pose landmarks
  MP_ASSERT_OK_AND_ASSIGN(auto pose_landmarks,
                          FetchResult<NormalizedLandmarkList>(
                              output_packets, kPoseLandmarksStream));
  EXPECT_THAT(
      pose_landmarks,
      Approximately(Partially(EqualsProto(holistic_result.pose_landmarks())),
                    /*margin=*/kAbsMargin));

  // Check Hand landmarks
  if (GetParam().holistic_request.is_left_hand_requested) {
    MP_ASSERT_OK_AND_ASSIGN(auto left_hand_landmarks,
                            FetchResult<NormalizedLandmarkList>(
                                output_packets, kLeftHandLandmarksStream));
    EXPECT_THAT(
        left_hand_landmarks,
        Approximately(
            Partially(EqualsProto(holistic_result.left_hand_landmarks())),
            /*margin=*/kAbsMargin));
  } else {
    ASSERT_FALSE(
        output_packets.contains(std::string(kLeftHandLandmarksStream)));
  }

  if (GetParam().holistic_request.is_right_hand_requested) {
    MP_ASSERT_OK_AND_ASSIGN(auto right_hand_landmarks,
                            FetchResult<NormalizedLandmarkList>(
                                output_packets, kRightHandLandmarksStream));
    EXPECT_THAT(
        right_hand_landmarks,
        Approximately(
            Partially(EqualsProto(holistic_result.right_hand_landmarks())),
            /*margin=*/kAbsMargin));
  } else {
    ASSERT_FALSE(
        output_packets.contains(std::string(kRightHandLandmarksStream)));
  }

  auto rendered_image =
      output_packets.at(std::string(kRenderedImageOutStream)).Get<Image>();
  MP_EXPECT_OK(SavePngTestOutput(
      *rendered_image.GetImageFrameSharedPtr(),
      absl::StrCat("holistic_landmark_",
                   RequestToString(GetParam().holistic_request))));

  auto pose_segmentation_mask =
      output_packets.at(std::string(kPoseSegmentationMaskStream)).Get<Image>();

  cv::Mat matting_mask = mediapipe::formats::MatView(
      pose_segmentation_mask.GetImageFrameSharedPtr().get());
  cv::Mat visualized_mask;
  matting_mask.convertTo(visualized_mask, CV_8UC1, 255);
  ImageFrame visualized_image(mediapipe::ImageFormat::GRAY8,
                              visualized_mask.cols, visualized_mask.rows,
                              visualized_mask.step, visualized_mask.data,
                              [visualized_mask](uint8_t[]) {});

  MP_EXPECT_OK(
      SavePngTestOutput(visualized_image, "holistic_pose_segmentation_mask"));
}

INSTANTIATE_TEST_SUITE_P(
    HolisticLandmarkerGraphTest, SmokeTest,
    Values(TestParams{
               /* test_name= */ "UseModelBundle",
               /* test_image_name= */ std::string(kTestImageFile),
               /* use_model_bundle= */ true,
               /* holistic_request= */
               {
                   /*is_left_hand_requested= */ true,
                   /*is_right_hand_requested= */ true,
                   /*is_face_requested= */ true,
                   /*is_face_blendshapes_requested= */ true,
               },
           },
           TestParams{
               /* test_name= */ "UseSeparateModelFiles",
               /* test_image_name= */ std::string(kTestImageFile),
               /* use_model_bundle= */ false,
               /* holistic_request= */
               {
                   /*is_left_hand_requested= */ true,
                   /*is_right_hand_requested= */ true,
                   /*is_face_requested= */ true,
                   /*is_face_blendshapes_requested= */ true,
               },
           },
           TestParams{
               /* test_name= */ "ModelBundleNoLeftHand",
               /* test_image_name= */ std::string(kTestImageFile),
               /* use_model_bundle= */ true,
               /* holistic_request= */
               {
                   /*is_left_hand_requested= */ false,
                   /*is_right_hand_requested= */ true,
                   /*is_face_requested= */ true,
                   /*is_face_blendshapes_requested= */ true,
               },
           },
           TestParams{
               /* test_name= */ "ModelBundleNoRightHand",
               /* test_image_name= */ std::string(kTestImageFile),
               /* use_model_bundle= */ true,
               /* holistic_request= */
               {
                   /*is_left_hand_requested= */ true,
                   /*is_right_hand_requested= */ false,
                   /*is_face_requested= */ true,
                   /*is_face_blendshapes_requested= */ true,
               },
           },
           TestParams{
               /* test_name= */ "ModelBundleNoHand",
               /* test_image_name= */ std::string(kTestImageFile),
               /* use_model_bundle= */ true,
               /* holistic_request= */
               {
                   /*is_left_hand_requested= */ false,
                   /*is_right_hand_requested= */ false,
                   /*is_face_requested= */ true,
                   /*is_face_blendshapes_requested= */ true,
               },
           },
           TestParams{
               /* test_name= */ "ModelBundleNoFace",
               /* test_image_name= */ std::string(kTestImageFile),
               /* use_model_bundle= */ true,
               /* holistic_request= */
               {
                   /*is_left_hand_requested= */ true,
                   /*is_right_hand_requested= */ true,
                   /*is_face_requested= */ false,
                   /*is_face_blendshapes_requested= */ false,
               },
           },
           TestParams{
               /* test_name= */ "ModelBundleNoFaceBlendshapes",
               /* test_image_name= */ std::string(kTestImageFile),
               /* use_model_bundle= */ true,
               /* holistic_request= */
               {
                   /*is_left_hand_requested= */ true,
                   /*is_right_hand_requested= */ true,
                   /*is_face_requested= */ true,
                   /*is_face_blendshapes_requested= */ false,
               },
           }),
    [](const TestParamInfo<SmokeTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
