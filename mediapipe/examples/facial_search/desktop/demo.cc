// Copyright 2020 The MediaPipe Authors.
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
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"

constexpr char kWindowName[] = "Find memes that match your facial expression";

#include "cpu_gpu_compatibility.h"
#include "mediapipe/examples/facial_search/embeddings.h"
#include "mediapipe/examples/facial_search/labels.h"

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_bool(log_embeddings, false, "Print embeddings vector in the log.");
DEFINE_bool(without_window, false, "Do not setup opencv window.");
DEFINE_string(images_folder_path, "", "Full path of images directory.");

::mediapipe::Status RunMPPGraph() {
  std::string pbtxt;
  MP_RETURN_IF_ERROR(
      mediapipe::file::GetContents(FLAGS_calculator_graph_config_file, &pbtxt));
  LOG(INFO) << "Get calculator graph config contents: " << pbtxt;
  auto config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(pbtxt);
  LOG(INFO) << "Initialize the calculator graph.";
  const auto labels = ::mediapipe::MyCollectionLabels();
  const auto collection = ::mediapipe::MyEmbeddingsCollection();
  std::map<std::string, ::mediapipe::Packet> input_side_packets = {
      {"collection_labels", ::mediapipe::MakePacket<decltype(labels)>(labels)},
      {"embeddings_collection",
       ::mediapipe::MakePacket<decltype(collection)>(collection)},
  };
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
  MAYBE_INIT_GPU(graph);

  LOG(INFO) << "Load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video)
    capture.open(FLAGS_input_video_path);
  else
    capture.open(0);
  RET_CHECK(capture.isOpened());
  const double invCaptureFPS = 1.0 / capture.get(cv::CAP_PROP_FPS);
  if (!FLAGS_without_window) {
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    RET_CHECK(!FLAGS_images_folder_path.empty());
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerForEmbeddings,
                   graph.AddOutputStreamPoller("embeddings"));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerForEmbeddingsPresence,
                   graph.AddOutputStreamPoller("embeddings_presence"));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerForMemes,
                   graph.AddOutputStreamPoller("memes"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  for (size_t n = 0; true; ++n) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame;
    capture >> camera_frame;
    if (camera_frame.empty()) break;  // End of video.
    if (!load_video)
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    if (!FLAGS_without_window) {
      cv::imshow("You", camera_frame);
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) continue;
    }
    cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGBA, 4);
    const auto ts = mediapipe::Timestamp::FromSeconds(n * invCaptureFPS);
    LOG(INFO) << "ts = " << ts;
    ADD_INPUT_FRAME("input_frame", camera_frame, ts);

    mediapipe::Packet presence;
    if (!pollerForEmbeddingsPresence.Next(&presence)) break;
    if (!presence.Get<bool>()) continue;

    if (FLAGS_log_embeddings) {
      LOG(INFO) << "polling for embeddings";
      mediapipe::Packet packet;
      if (!pollerForEmbeddings.Next(&packet)) break;
      const auto& embedding = packet.Get<std::vector<float>>();
      std::ostringstream oss;
      oss << "{";
      for (float f : embedding) oss << f << ",";
      oss << "},\n";
      LOG(INFO) << oss.str();
    }

    LOG(INFO) << "polling for memes";
    mediapipe::Packet packet;
    if (!pollerForMemes.Next(&packet)) break;
    const auto& memes = packet.Get<std::vector<mediapipe::Classification>>();
    LOG(INFO) << "#memes: " << memes.size();
    for (const auto& meme : memes) {
      LOG(INFO) << meme.score() << " <-- " << meme.label();
    }

    if (!FLAGS_without_window && !memes.empty()) {
      const auto img_path = FLAGS_images_folder_path + memes[0].label();
      const auto image = cv::imread(img_path, cv::IMREAD_UNCHANGED);
      RET_CHECK(image.data) << "Couldn't load " << img_path;
      cv::imshow(kWindowName, image);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) break;
    }
  }

  LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
  } else {
    LOG(INFO) << "Success!";
  }
  return 0;
}
