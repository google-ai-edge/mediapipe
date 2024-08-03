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
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

// Take stream from /mediapipe/graphs/hand_tracking/hand_detection_desktop_live.pbtxt
//  RendererSubgraph - LANDMARKS:hand_landmarks
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

// 这些语句定义了一系列的C++常量表达式。
// input and output streams to be used/retrieved by calculators
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kPoseLandmarksStream[] = "pose_landmarks";
constexpr char kLeftHandLandmarksStream[] = "left_hand_landmarks";
constexpr char kRightHandLandmarksStream[] = "right_hand_landmarks";
constexpr char kFaceLandmarksStream[] = "face_landmarks";
constexpr char kWindowName[] = "MediaPipe";

// 定义命令行参数，用于指定MediaPipe图配置文件的位置
// cli inputs
DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

// MediaPipe框架中的一个函数
::mediapipe::Status RunMPPGraph()
{
    // 读取并解析计算器图配置。
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: "
              << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    // 在使用MediaPipe框架中初始化一个计算器图。
    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    // 初始化摄像头
    LOG(INFO) << "Initialize the camera.";
    cv::VideoCapture capture;
    capture.open(0);
    RET_CHECK(capture.isOpened());
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);

    // pollers to retrieve streams from graph
    // output stream (i.e. rendered landmark frame)

    // 初始化了MediaPipe计算图的执行，并设置了用于从指定输出流中提取数据的`OutputStreamPoller`。
    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller poller,
                     graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller pose_poller_landmark,
                     graph.AddOutputStreamPoller(kPoseLandmarksStream));
    ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller face_poller_landmark,
                     graph.AddOutputStreamPoller(kFaceLandmarksStream));
    ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller left_hand_poller_landmark,
                     graph.AddOutputStreamPoller(kLeftHandLandmarksStream));
    ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller right_hand_poller_landmark,
                     graph.AddOutputStreamPoller(kRightHandLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    // 捕获摄像头的每一帧，转换颜色格式，并在需要时进行水平翻转
    LOG(INFO) << "Start grabbing and processing frames.";
    size_t frame_timestamp = 0;
    bool grab_frames = true;
    while (grab_frames)
    {
        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty())
            break; // End of video.
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

        // 将OpenCV的`cv::Mat`格式的帧转换为MediaPipe的`ImageFrame`格式。
        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // 负责将一个图像帧发送到MediaPipe的计算图中进行处理。
        // Send image packet into the graph.
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release())
                              .At(mediapipe::Timestamp(frame_timestamp++))));

        // 从MediaPipe图的输出流中获取处理后的图像帧，并将其存储在`output_frame`中。
        // Get the graph result packet, or stop if that fails.
        ::mediapipe::Packet packet;
        if (!poller.Next(&packet))
            break;

        // Use packet.Get to recover values from packet
        auto &output_frame = packet.Get<::mediapipe::ImageFrame>();

        // 从MediaPipe图的输出流中提取手部标记，并将其存储在`multi_hand_landmarks`变量中。
        // Get the packet containing multi_hand_landmarks.
        ::mediapipe::Packet pose_landmarks_packet;
        ::mediapipe::Packet face_landmarks_packet;
        ::mediapipe::Packet left_hand_landmarks_packet;
        ::mediapipe::Packet right_hand_landmarks_packet;

        if (!pose_poller_landmark.Next(&pose_landmarks_packet))
            break;
        if (!face_poller_landmark.Next(&face_landmarks_packet))
            break;
        if (!left_hand_poller_landmark.Next(&left_hand_landmarks_packet))
            break;
        if (!right_hand_poller_landmark.Next(&right_hand_landmarks_packet))
            break;

        const auto &pose_landmarks = pose_landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();
        const auto &face_landmarks = face_landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();
        const auto &left_hand_landmarks = left_hand_landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();
        const auto &right_hand_landmarks = right_hand_landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();

        LOG(INFO) << "#Pose landmarks: " << pose_landmarks.landmark_size();
        for (int i = 0; i < pose_landmarks.landmark_size(); ++i)
        {
            const auto &landmark = pose_landmarks.landmark(i);
            LOG(INFO) << "\tPose Landmark [" << i << "]: ("
                      << landmark.x() << ", "
                      << landmark.y() << ", "
                      << landmark.z() << ")";
        }

        LOG(INFO) << "#Face landmarks: " << face_landmarks.landmark_size();
        for (int i = 0; i < face_landmarks.landmark_size(); ++i)
        {
            const auto &landmark = face_landmarks.landmark(i);
            LOG(INFO) << "\tFace Landmark [" << i << "]: ("
                      << landmark.x() << ", "
                      << landmark.y() << ", "
                      << landmark.z() << ")";
        }

        LOG(INFO) << "#Left Hand landmarks: " << left_hand_landmarks.landmark_size();
        for (int i = 0; i < left_hand_landmarks.landmark_size(); ++i)
        {
            const auto &landmark = left_hand_landmarks.landmark(i);
            LOG(INFO) << "\tLeft Hand Landmark [" << i << "]: ("
                      << landmark.x() << ", "
                      << landmark.y() << ", "
                      << landmark.z() << ")";
        }

        LOG(INFO) << "#Right Hand landmarks: " << right_hand_landmarks.landmark_size();
        for (int i = 0; i < right_hand_landmarks.landmark_size(); ++i)
        {
            const auto &landmark = right_hand_landmarks.landmark(i);
            LOG(INFO) << "\tFace Landmark [" << i << "]: ("
                      << landmark.x() << ", "
                      << landmark.y() << ", "
                      << landmark.z() << ")";
        }
        // Use packet.Get to recover values from packet

        // 使用OpenCV和MediaPipe进行图像处理和显示。
        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);

        // 这一行会等待5毫秒以查看用户是否按下了任何键。
        // Press any key to exit.
        const int pressed_key = cv::waitKey(5);
        if (pressed_key >= 0 && pressed_key != 255)
            grab_frames = false;
    }

    // 使用MediaPipe框架和OpenCV库在处理视频数据后的关闭和清理步骤。
    LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

// 程序的主入口点，也就是`main`函数。它描述了一个使用MediaPipe框架的程序如何初始化，执行，并处理结果。
int main(int argc, char **argv)
{
    // **初始化 Google 日志**:
    google::InitGoogleLogging(argv[0]);

    // **解析命令行参数**:
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // **运行 MediaPipe 图**:
    ::mediapipe::Status run_status = RunMPPGraph();

    // **处理结果**:
    if (!run_status.ok())
    {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    }
    else
    {
        LOG(INFO) << "Success!";
    }
    return 0;
}