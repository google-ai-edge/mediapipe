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

#include <cmath>
#include <fstream>
#include <memory>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/video/motion_analysis_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/tracking/camera_motion.h"
#include "mediapipe/util/tracking/camera_motion.pb.h"
#include "mediapipe/util/tracking/frame_selection.pb.h"
#include "mediapipe/util/tracking/motion_analysis.h"
#include "mediapipe/util/tracking/motion_estimation.h"
#include "mediapipe/util/tracking/motion_models.h"
#include "mediapipe/util/tracking/region_flow.pb.h"

namespace mediapipe {

using mediapipe::AffineAdapter;
using mediapipe::CameraMotion;
using mediapipe::FrameSelectionResult;
using mediapipe::Homography;
using mediapipe::HomographyAdapter;
using mediapipe::LinearSimilarityModel;
using mediapipe::MixtureHomography;
using mediapipe::MixtureRowWeights;
using mediapipe::MotionAnalysis;
using mediapipe::ProjectViaFit;
using mediapipe::RegionFlowComputationOptions;
using mediapipe::RegionFlowFeatureList;
using mediapipe::SalientPointFrame;
using mediapipe::TranslationModel;

const char kOptionsTag[] = "OPTIONS";

// A calculator that performs motion analysis on an incoming video stream.
//
// Input streams:  (at least one of them is required).
//   VIDEO:     The input video stream (ImageFrame, sRGB, sRGBA or GRAY8).
//   SELECTION: Optional input stream to perform analysis only on selected
//              frames. If present needs to contain camera motion
//              and features.
//
// Input side packets:
//   CSV_FILE:  Read motion models as homographies from CSV file. Expected
//              to be defined in the frame domain (un-normalized).
//              Should store 9 floats per row.
//              Specify number of homographies per frames via option
//              meta_models_per_frame. For values > 1, MixtureHomographies
//              are created, for value == 1, a single Homography is used.
//   DOWNSAMPLE: Optionally specify downsampling factor via input side packet
//               overriding value in the graph settings.
// Output streams (all are optional).
//   FLOW:      Sparse feature tracks in form of proto RegionFlowFeatureList.
//   CAMERA:    Camera motion as proto CameraMotion describing the per frame-
//              pair motion. Has VideoHeader from input video.
//   SALIENCY:  Foreground saliency (objects moving different from the
//              background) as proto SalientPointFrame.
//   VIZ:       Visualization stream as ImageFrame, sRGB, visualizing
//              features and saliency (set via
//              analysis_options().visualization_options())
//   DENSE_FG:  Dense foreground stream, describing per-pixel foreground-
//              ness as confidence between 0 (background) and 255
//              (foreground). Output is ImageFrame (GRAY8).
//   VIDEO_OUT: Optional output stream when SELECTION is used. Output is input
//              VIDEO at the selected frames. Required VIDEO to be present.
//   GRAY_VIDEO_OUT: Optional output stream for downsampled, grayscale video.
//                   Requires VIDEO to be present and SELECTION to not be used.
class MotionAnalysisCalculator : public CalculatorBase {
  // TODO: Activate once leakr approval is ready.
  // typedef com::google::android::libraries::micro::proto::Data HomographyData;

 public:
  ~MotionAnalysisCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  // Outputs results to Outputs() if MotionAnalysis buffered sufficient results.
  // Otherwise no-op. Set flush to true to force output of all buffered data.
  void OutputMotionAnalyzedFrames(bool flush, CalculatorContext* cc);

  // Lazy init function to be called on Process.
  ::mediapipe::Status InitOnProcess(InputStream* video_stream,
                                    InputStream* selection_stream);

  // Parses CSV file contents to homographies.
  bool ParseModelCSV(const std::string& contents,
                     std::deque<Homography>* homographies);

  // Turns list of 9-tuple floating values into set of homographies.
  bool HomographiesFromValues(const std::vector<float>& homog_values,
                              std::deque<Homography>* homographies);

  // Appends CameraMotions and features from homographies.
  // Set append_identity to true to add an identity transform to the beginning
  // of the each list *in addition* to the motions derived from homographies.
  void AppendCameraMotionsFromHomographies(
      const std::deque<Homography>& homographies, bool append_identity,
      std::deque<CameraMotion>* camera_motions,
      std::deque<RegionFlowFeatureList>* features);

  // Helper function to subtract current metadata motion from features. Used
  // for hybrid estimation case.
  void SubtractMetaMotion(const CameraMotion& meta_motion,
                          RegionFlowFeatureList* features);

  // Inverse of above function to add back meta motion and replace
  // feature location with originals after estimation.
  void AddMetaMotion(const CameraMotion& meta_motion,
                     const RegionFlowFeatureList& meta_features,
                     RegionFlowFeatureList* features, CameraMotion* motion);

  MotionAnalysisCalculatorOptions options_;
  int frame_width_ = -1;
  int frame_height_ = -1;
  int frame_idx_ = 0;

  // Buffers incoming video frame packets (if visualization output is requested)
  std::vector<Packet> packet_buffer_;

  // Buffers incoming timestamps until MotionAnalysis is ready to output via
  // above OutputMotionAnalyzedFrames.
  std::vector<Timestamp> timestamp_buffer_;

  // Input indicators for each stream.
  bool selection_input_ = false;
  bool video_input_ = false;

  // Output indicators for each stream.
  bool region_flow_feature_output_ = false;
  bool camera_motion_output_ = false;
  bool saliency_output_ = false;
  bool visualize_output_ = false;
  bool dense_foreground_output_ = false;
  bool video_output_ = false;
  bool grayscale_output_ = false;
  bool csv_file_input_ = false;

  // Inidicates if saliency should be computed.
  bool with_saliency_ = false;

  // Set if hybrid meta analysis - see proto for details.
  bool hybrid_meta_analysis_ = false;

  // Concatenated motions for each selected frame. Used in case
  // hybrid estimation is requested to fallback to valid models.
  std::deque<CameraMotion> selected_motions_;

  // Normalized homographies from CSV file or metadata.
  std::deque<Homography> meta_homographies_;
  std::deque<CameraMotion> meta_motions_;
  std::deque<RegionFlowFeatureList> meta_features_;

  // Offset into above meta_motions_ and features_ when using
  // hybrid meta analysis.
  int hybrid_meta_offset_ = 0;

  std::unique_ptr<MotionAnalysis> motion_analysis_;

  std::unique_ptr<MixtureRowWeights> row_weights_;
};

REGISTER_CALCULATOR(MotionAnalysisCalculator);

::mediapipe::Status MotionAnalysisCalculator::GetContract(
    CalculatorContract* cc) {
  if (cc->Inputs().HasTag("VIDEO")) {
    cc->Inputs().Tag("VIDEO").Set<ImageFrame>();
  }

  // Optional input stream from frame selection calculator.
  if (cc->Inputs().HasTag("SELECTION")) {
    cc->Inputs().Tag("SELECTION").Set<FrameSelectionResult>();
  }

  RET_CHECK(cc->Inputs().HasTag("VIDEO") || cc->Inputs().HasTag("SELECTION"))
      << "Either VIDEO, SELECTION must be specified.";

  if (cc->Outputs().HasTag("FLOW")) {
    cc->Outputs().Tag("FLOW").Set<RegionFlowFeatureList>();
  }

  if (cc->Outputs().HasTag("CAMERA")) {
    cc->Outputs().Tag("CAMERA").Set<CameraMotion>();
  }

  if (cc->Outputs().HasTag("SALIENCY")) {
    cc->Outputs().Tag("SALIENCY").Set<SalientPointFrame>();
  }

  if (cc->Outputs().HasTag("VIZ")) {
    cc->Outputs().Tag("VIZ").Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag("DENSE_FG")) {
    cc->Outputs().Tag("DENSE_FG").Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag("VIDEO_OUT")) {
    cc->Outputs().Tag("VIDEO_OUT").Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag("GRAY_VIDEO_OUT")) {
    // We only output grayscale video if we're actually performing full region-
    // flow analysis on the video.
    RET_CHECK(cc->Inputs().HasTag("VIDEO") &&
              !cc->Inputs().HasTag("SELECTION"));
    cc->Outputs().Tag("GRAY_VIDEO_OUT").Set<ImageFrame>();
  }

  if (cc->InputSidePackets().HasTag("CSV_FILE")) {
    cc->InputSidePackets().Tag("CSV_FILE").Set<std::string>();
  }
  if (cc->InputSidePackets().HasTag("DOWNSAMPLE")) {
    cc->InputSidePackets().Tag("DOWNSAMPLE").Set<float>();
  }

  if (cc->InputSidePackets().HasTag(kOptionsTag)) {
    cc->InputSidePackets().Tag(kOptionsTag).Set<CalculatorOptions>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status MotionAnalysisCalculator::Open(CalculatorContext* cc) {
  options_ =
      tool::RetrieveOptions(cc->Options<MotionAnalysisCalculatorOptions>(),
                            cc->InputSidePackets(), kOptionsTag);

  video_input_ = cc->Inputs().HasTag("VIDEO");
  selection_input_ = cc->Inputs().HasTag("SELECTION");
  region_flow_feature_output_ = cc->Outputs().HasTag("FLOW");
  camera_motion_output_ = cc->Outputs().HasTag("CAMERA");
  saliency_output_ = cc->Outputs().HasTag("SALIENCY");
  visualize_output_ = cc->Outputs().HasTag("VIZ");
  dense_foreground_output_ = cc->Outputs().HasTag("DENSE_FG");
  video_output_ = cc->Outputs().HasTag("VIDEO_OUT");
  grayscale_output_ = cc->Outputs().HasTag("GRAY_VIDEO_OUT");
  csv_file_input_ = cc->InputSidePackets().HasTag("CSV_FILE");
  hybrid_meta_analysis_ = options_.meta_analysis() ==
                          MotionAnalysisCalculatorOptions::META_ANALYSIS_HYBRID;

  if (video_output_) {
    RET_CHECK(selection_input_) << "VIDEO_OUT requires SELECTION input";
  }

  if (selection_input_) {
    switch (options_.selection_analysis()) {
      case MotionAnalysisCalculatorOptions::NO_ANALYSIS_USE_SELECTION:
        RET_CHECK(!visualize_output_)
            << "Visualization not supported for NO_ANALYSIS_USE_SELECTION";
        RET_CHECK(!dense_foreground_output_)
            << "Dense foreground not supported for NO_ANALYSIS_USE_SELECTION";
        RET_CHECK(!saliency_output_)
            << "Saliency output not supported for NO_ANALYSIS_USE_SELECTION";
        break;

      case MotionAnalysisCalculatorOptions::ANALYSIS_RECOMPUTE:
      case MotionAnalysisCalculatorOptions::ANALYSIS_WITH_SEED:
        RET_CHECK(video_input_) << "Need video input for feature tracking.";
        break;

      case MotionAnalysisCalculatorOptions::ANALYSIS_FROM_FEATURES:
        // Nothing to add here.
        break;
    }
  }

  if (visualize_output_ || dense_foreground_output_ || video_output_) {
    RET_CHECK(video_input_) << "Video input required.";
  }

  if (csv_file_input_) {
    RET_CHECK(!selection_input_)
        << "Can not use selection input with csv input.";
    if (!hybrid_meta_analysis_) {
      RET_CHECK(!saliency_output_ && !visualize_output_ &&
                !dense_foreground_output_ && !grayscale_output_)
          << "CSV file and meta input only supports flow and camera motion "
          << "output when using metadata only.";
    }
  }

  if (csv_file_input_) {
    // Read from file and parse.
    const std::string filename =
        cc->InputSidePackets().Tag("CSV_FILE").Get<std::string>();

    std::string file_contents;
    std::ifstream input_file(filename, std::ios::in);
    input_file.seekg(0, std::ios::end);
    const int file_length = input_file.tellg();
    file_contents.resize(file_length);
    input_file.seekg(0, std::ios::beg);
    input_file.read(&file_contents[0], file_length);
    input_file.close();

    RET_CHECK(ParseModelCSV(file_contents, &meta_homographies_))
        << "Could not parse CSV file";
  }

  // Get video header from video or selection input if present.
  const VideoHeader* video_header = nullptr;
  if (video_input_ && !cc->Inputs().Tag("VIDEO").Header().IsEmpty()) {
    video_header = &(cc->Inputs().Tag("VIDEO").Header().Get<VideoHeader>());
  } else if (selection_input_ &&
             !cc->Inputs().Tag("SELECTION").Header().IsEmpty()) {
    video_header = &(cc->Inputs().Tag("SELECTION").Header().Get<VideoHeader>());
  } else {
    LOG(WARNING) << "No input video header found. Downstream calculators "
                    "expecting video headers are likely to fail.";
  }

  with_saliency_ = options_.analysis_options().compute_motion_saliency();
  // Force computation of saliency if requested as output.
  if (cc->Outputs().HasTag("SALIENCY")) {
    with_saliency_ = true;
    if (!options_.analysis_options().compute_motion_saliency()) {
      LOG(WARNING) << "Enable saliency computation. Set "
                   << "compute_motion_saliency to true to silence this "
                   << "warning.";
      options_.mutable_analysis_options()->set_compute_motion_saliency(true);
    }
  }

  if (options_.bypass_mode()) {
    cc->SetOffset(TimestampDiff(0));
  }

  if (cc->InputSidePackets().HasTag("DOWNSAMPLE")) {
    options_.mutable_analysis_options()
        ->mutable_flow_options()
        ->set_downsample_factor(
            cc->InputSidePackets().Tag("DOWNSAMPLE").Get<float>());
  }

  // If no video header is provided, just return and initialize on the first
  // Process() call.
  if (video_header == nullptr) {
    return ::mediapipe::OkStatus();
  }

  ////////////// EARLY RETURN; ONLY HEADER OUTPUT SHOULD GO HERE ///////////////

  if (visualize_output_) {
    cc->Outputs().Tag("VIZ").SetHeader(Adopt(new VideoHeader(*video_header)));
  }

  if (video_output_) {
    cc->Outputs()
        .Tag("VIDEO_OUT")
        .SetHeader(Adopt(new VideoHeader(*video_header)));
  }

  if (cc->Outputs().HasTag("DENSE_FG")) {
    std::unique_ptr<VideoHeader> foreground_header(
        new VideoHeader(*video_header));
    foreground_header->format = ImageFormat::GRAY8;
    cc->Outputs().Tag("DENSE_FG").SetHeader(Adopt(foreground_header.release()));
  }

  if (cc->Outputs().HasTag("CAMERA")) {
    cc->Outputs().Tag("CAMERA").SetHeader(
        Adopt(new VideoHeader(*video_header)));
  }

  if (cc->Outputs().HasTag("SALIENCY")) {
    cc->Outputs()
        .Tag("SALIENCY")
        .SetHeader(Adopt(new VideoHeader(*video_header)));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status MotionAnalysisCalculator::Process(CalculatorContext* cc) {
  if (options_.bypass_mode()) {
    return ::mediapipe::OkStatus();
  }

  InputStream* video_stream =
      video_input_ ? &(cc->Inputs().Tag("VIDEO")) : nullptr;
  InputStream* selection_stream =
      selection_input_ ? &(cc->Inputs().Tag("SELECTION")) : nullptr;

  // Checked on Open.
  CHECK(video_stream || selection_stream);

  // Lazy init.
  if (frame_width_ < 0 || frame_height_ < 0) {
    MP_RETURN_IF_ERROR(InitOnProcess(video_stream, selection_stream));
  }

  const Timestamp timestamp = cc->InputTimestamp();
  if ((csv_file_input_) && !hybrid_meta_analysis_) {
    if (camera_motion_output_) {
      RET_CHECK(!meta_motions_.empty()) << "Insufficient metadata.";

      CameraMotion output_motion = meta_motions_.front();
      meta_motions_.pop_front();
      output_motion.set_timestamp_usec(timestamp.Value());
      cc->Outputs().Tag("CAMERA").Add(new CameraMotion(output_motion),
                                      timestamp);
    }

    if (region_flow_feature_output_) {
      RET_CHECK(!meta_features_.empty()) << "Insufficient frames in CSV file";
      RegionFlowFeatureList output_features = meta_features_.front();
      meta_features_.pop_front();

      output_features.set_timestamp_usec(timestamp.Value());
      cc->Outputs().Tag("FLOW").Add(new RegionFlowFeatureList(output_features),
                                    timestamp);
    }

    ++frame_idx_;
    return ::mediapipe::OkStatus();
  }

  if (motion_analysis_ == nullptr) {
    // We do not need MotionAnalysis when using just metadata.
    motion_analysis_.reset(new MotionAnalysis(options_.analysis_options(),
                                              frame_width_, frame_height_));
  }

  std::unique_ptr<FrameSelectionResult> frame_selection_result;
  // Always use frame if selection is not activated.
  bool use_frame = !selection_input_;
  if (selection_input_) {
    CHECK(selection_stream);

    // Fill in timestamps we process.
    if (!selection_stream->Value().IsEmpty()) {
      ASSIGN_OR_RETURN(
          frame_selection_result,
          selection_stream->Value().ConsumeOrCopy<FrameSelectionResult>());
      use_frame = true;

      // Make sure both features and camera motion are present.
      RET_CHECK(frame_selection_result->has_camera_motion() &&
                frame_selection_result->has_features())
          << "Frame selection input error at: " << timestamp
          << " both camera motion and features need to be "
             "present in FrameSelectionResult. "
          << frame_selection_result->has_camera_motion() << " , "
          << frame_selection_result->has_features();
    }
  }

  if (selection_input_ && use_frame &&
      options_.selection_analysis() ==
          MotionAnalysisCalculatorOptions::NO_ANALYSIS_USE_SELECTION) {
    // Output concatenated results, nothing to compute here.
    if (camera_motion_output_) {
      cc->Outputs().Tag("CAMERA").Add(
          frame_selection_result->release_camera_motion(), timestamp);
    }
    if (region_flow_feature_output_) {
      cc->Outputs().Tag("FLOW").Add(frame_selection_result->release_features(),
                                    timestamp);
    }

    if (video_output_) {
      cc->Outputs().Tag("VIDEO_OUT").AddPacket(video_stream->Value());
    }

    return ::mediapipe::OkStatus();
  }

  if (use_frame) {
    if (!selection_input_) {
      const cv::Mat input_view =
          formats::MatView(&video_stream->Get<ImageFrame>());
      if (hybrid_meta_analysis_) {
        // Seed with meta homography.
        RET_CHECK(hybrid_meta_offset_ < meta_motions_.size())
            << "Not enough metadata received for hybrid meta analysis";
        Homography initial_transform =
            meta_motions_[hybrid_meta_offset_].homography();
        std::function<void(RegionFlowFeatureList*)> subtract_helper = std::bind(
            &MotionAnalysisCalculator::SubtractMetaMotion, this,
            meta_motions_[hybrid_meta_offset_], std::placeholders::_1);

        // Keep original features before modification around.
        motion_analysis_->AddFrameGeneric(
            input_view, timestamp.Value(), initial_transform, nullptr, nullptr,
            &subtract_helper, &meta_features_[hybrid_meta_offset_]);
        ++hybrid_meta_offset_;
      } else {
        motion_analysis_->AddFrame(input_view, timestamp.Value());
      }
    } else {
      selected_motions_.push_back(frame_selection_result->camera_motion());
      switch (options_.selection_analysis()) {
        case MotionAnalysisCalculatorOptions::NO_ANALYSIS_USE_SELECTION:
          return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
                 << "Should not reach this point!";

        case MotionAnalysisCalculatorOptions::ANALYSIS_FROM_FEATURES:
          motion_analysis_->AddFeatures(frame_selection_result->features());
          break;

        case MotionAnalysisCalculatorOptions::ANALYSIS_RECOMPUTE: {
          const cv::Mat input_view =
              formats::MatView(&video_stream->Get<ImageFrame>());
          motion_analysis_->AddFrame(input_view, timestamp.Value());
          break;
        }

        case MotionAnalysisCalculatorOptions::ANALYSIS_WITH_SEED: {
          Homography homography;
          CameraMotionToHomography(frame_selection_result->camera_motion(),
                                   &homography);
          const cv::Mat input_view =
              formats::MatView(&video_stream->Get<ImageFrame>());
          motion_analysis_->AddFrameGeneric(input_view, timestamp.Value(),
                                            homography, &homography);
          break;
        }
      }
    }

    timestamp_buffer_.push_back(timestamp);
    ++frame_idx_;

    VLOG_EVERY_N(0, 100) << "Analyzed frame " << frame_idx_;

    // Buffer input frames only if visualization is requested.
    if (visualize_output_ || video_output_) {
      packet_buffer_.push_back(video_stream->Value());
    }

    // If requested, output grayscale thumbnails
    if (grayscale_output_) {
      cv::Mat grayscale_mat = motion_analysis_->GetGrayscaleFrameFromResults();
      std::unique_ptr<ImageFrame> grayscale_image(new ImageFrame(
          ImageFormat::GRAY8, grayscale_mat.cols, grayscale_mat.rows));
      cv::Mat image_frame_mat = formats::MatView(grayscale_image.get());
      grayscale_mat.copyTo(image_frame_mat);

      cc->Outputs()
          .Tag("GRAY_VIDEO_OUT")
          .Add(grayscale_image.release(), timestamp);
    }

    // Output other results, if we have any yet.
    OutputMotionAnalyzedFrames(false, cc);
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status MotionAnalysisCalculator::Close(CalculatorContext* cc) {
  // Guard against empty videos.
  if (motion_analysis_) {
    OutputMotionAnalyzedFrames(true, cc);
  }
  if (csv_file_input_) {
    if (!meta_motions_.empty()) {
      LOG(ERROR) << "More motions than frames. Unexpected! Remainder: "
                 << meta_motions_.size();
    }
  }
  return ::mediapipe::OkStatus();
}

void MotionAnalysisCalculator::OutputMotionAnalyzedFrames(
    bool flush, CalculatorContext* cc) {
  std::vector<std::unique_ptr<RegionFlowFeatureList>> features;
  std::vector<std::unique_ptr<CameraMotion>> camera_motions;
  std::vector<std::unique_ptr<SalientPointFrame>> saliency;

  const int buffer_size = timestamp_buffer_.size();
  const int num_results = motion_analysis_->GetResults(
      flush, &features, &camera_motions, with_saliency_ ? &saliency : nullptr);

  CHECK_LE(num_results, buffer_size);

  if (num_results == 0) {
    return;
  }

  for (int k = 0; k < num_results; ++k) {
    // Region flow features and camera motion for this frame.
    auto& feature_list = features[k];
    auto& camera_motion = camera_motions[k];
    const Timestamp timestamp = timestamp_buffer_[k];

    if (selection_input_ && options_.hybrid_selection_camera()) {
      if (camera_motion->type() > selected_motions_.front().type()) {
        // Composited type is more stable.
        camera_motion->Swap(&selected_motions_.front());
      }
      selected_motions_.pop_front();
    }

    if (hybrid_meta_analysis_) {
      AddMetaMotion(meta_motions_.front(), meta_features_.front(),
                    feature_list.get(), camera_motion.get());
      meta_motions_.pop_front();
      meta_features_.pop_front();
    }

    // Video frame for visualization.
    std::unique_ptr<ImageFrame> visualization_frame;
    cv::Mat visualization;
    if (visualize_output_) {
      // Initialize visualization frame with original frame.
      visualization_frame.reset(new ImageFrame());
      visualization_frame->CopyFrom(packet_buffer_[k].Get<ImageFrame>(), 16);
      visualization = formats::MatView(visualization_frame.get());

      motion_analysis_->RenderResults(
          *feature_list, *camera_motion,
          with_saliency_ ? saliency[k].get() : nullptr, &visualization);

      cc->Outputs().Tag("VIZ").Add(visualization_frame.release(), timestamp);
    }

    // Output dense foreground mask.
    if (dense_foreground_output_) {
      std::unique_ptr<ImageFrame> foreground_frame(
          new ImageFrame(ImageFormat::GRAY8, frame_width_, frame_height_));
      cv::Mat foreground = formats::MatView(foreground_frame.get());
      motion_analysis_->ComputeDenseForeground(*feature_list, *camera_motion,
                                               &foreground);
      cc->Outputs().Tag("DENSE_FG").Add(foreground_frame.release(), timestamp);
    }

    // Output flow features if requested.
    if (region_flow_feature_output_) {
      cc->Outputs().Tag("FLOW").Add(feature_list.release(), timestamp);
    }

    // Output camera motion.
    if (camera_motion_output_) {
      cc->Outputs().Tag("CAMERA").Add(camera_motion.release(), timestamp);
    }

    if (video_output_) {
      cc->Outputs().Tag("VIDEO_OUT").AddPacket(packet_buffer_[k]);
    }

    // Output saliency.
    if (saliency_output_) {
      cc->Outputs().Tag("SALIENCY").Add(saliency[k].release(), timestamp);
    }
  }

  if (hybrid_meta_analysis_) {
    hybrid_meta_offset_ -= num_results;
    CHECK_GE(hybrid_meta_offset_, 0);
  }

  timestamp_buffer_.erase(timestamp_buffer_.begin(),
                          timestamp_buffer_.begin() + num_results);

  if (visualize_output_ || video_output_) {
    packet_buffer_.erase(packet_buffer_.begin(),
                         packet_buffer_.begin() + num_results);
  }
}

::mediapipe::Status MotionAnalysisCalculator::InitOnProcess(
    InputStream* video_stream, InputStream* selection_stream) {
  if (video_stream) {
    frame_width_ = video_stream->Get<ImageFrame>().Width();
    frame_height_ = video_stream->Get<ImageFrame>().Height();

    // Ensure image options are set correctly.
    auto* region_options =
        options_.mutable_analysis_options()->mutable_flow_options();

    // Use two possible formats to account for different channel orders.
    RegionFlowComputationOptions::ImageFormat image_format;
    RegionFlowComputationOptions::ImageFormat image_format2;
    switch (video_stream->Get<ImageFrame>().Format()) {
      case ImageFormat::GRAY8:
        image_format = image_format2 =
            RegionFlowComputationOptions::FORMAT_GRAYSCALE;
        break;

      case ImageFormat::SRGB:
        image_format = RegionFlowComputationOptions::FORMAT_RGB;
        image_format2 = RegionFlowComputationOptions::FORMAT_BGR;
        break;

      case ImageFormat::SRGBA:
        image_format = RegionFlowComputationOptions::FORMAT_RGBA;
        image_format2 = RegionFlowComputationOptions::FORMAT_BGRA;
        break;

      default:
        RET_CHECK(false) << "Unsupported image format.";
    }
    if (region_options->image_format() != image_format &&
        region_options->image_format() != image_format2) {
      LOG(WARNING) << "Requested image format in RegionFlowComputation "
                   << "does not match video stream format. Overriding.";
      region_options->set_image_format(image_format);
    }

    // Account for downsampling mode INPUT_SIZE. In this case we are handed
    // already downsampled frames but the resulting CameraMotion should
    // be computed on higher resolution as specifed by the downsample scale.
    if (region_options->downsample_mode() ==
        RegionFlowComputationOptions::DOWNSAMPLE_TO_INPUT_SIZE) {
      const float scale = region_options->downsample_factor();
      frame_width_ = static_cast<int>(std::round(frame_width_ * scale));
      frame_height_ = static_cast<int>(std::round(frame_height_ * scale));
    }
  } else if (selection_stream) {
    const auto& camera_motion =
        selection_stream->Get<FrameSelectionResult>().camera_motion();
    frame_width_ = camera_motion.frame_width();
    frame_height_ = camera_motion.frame_height();
  } else {
    LOG(FATAL) << "Either VIDEO or SELECTION stream need to be specified.";
  }

  // Filled by CSV file parsing.
  if (!meta_homographies_.empty()) {
    CHECK(csv_file_input_);
    AppendCameraMotionsFromHomographies(meta_homographies_,
                                        true,  // append identity.
                                        &meta_motions_, &meta_features_);
    meta_homographies_.clear();
  }

  // Filter weights before using for hybrid mode.
  if (hybrid_meta_analysis_) {
    auto* motion_options =
        options_.mutable_analysis_options()->mutable_motion_options();
    motion_options->set_filter_initialized_irls_weights(true);
  }

  return ::mediapipe::OkStatus();
}

bool MotionAnalysisCalculator::ParseModelCSV(
    const std::string& contents, std::deque<Homography>* homographies) {
  std::vector<absl::string_view> values =
      absl::StrSplit(contents, absl::ByAnyChar(",\n"));

  // Trim off any empty lines.
  while (values.back().empty()) {
    values.pop_back();
  }

  // Convert to float.
  std::vector<float> homog_values;
  homog_values.reserve(values.size());

  for (const auto& value : values) {
    double value_64f;
    if (!absl::SimpleAtod(value, &value_64f)) {
      LOG(ERROR) << "Not a double, expected!";
      return false;
    }

    homog_values.push_back(value_64f);
  }

  return HomographiesFromValues(homog_values, homographies);
}

bool MotionAnalysisCalculator::HomographiesFromValues(
    const std::vector<float>& homog_values,
    std::deque<Homography>* homographies) {
  CHECK(homographies);

  // Obvious constants are obvious :D
  constexpr int kHomographyValues = 9;
  if (homog_values.size() % kHomographyValues != 0) {
    LOG(ERROR) << "Contents not a multiple of " << kHomographyValues;
    return false;
  }

  for (int k = 0; k < homog_values.size(); k += kHomographyValues) {
    std::vector<double> h_vals(kHomographyValues);
    for (int l = 0; l < kHomographyValues; ++l) {
      h_vals[l] = homog_values[k + l];
    }

    // Normalize last entry to 1.
    if (h_vals[kHomographyValues - 1] == 0) {
      LOG(ERROR) << "Degenerate homography, last entry is zero";
      return false;
    }

    const double scale = 1.0f / h_vals[kHomographyValues - 1];
    for (int l = 0; l < kHomographyValues; ++l) {
      h_vals[l] *= scale;
    }

    Homography h = HomographyAdapter::FromDoublePointer(h_vals.data(), false);
    homographies->push_back(h);
  }

  if (homographies->size() % options_.meta_models_per_frame() != 0) {
    LOG(ERROR) << "Total homographies not a multiple of specified models "
               << "per frame.";
    return false;
  }

  return true;
}

void MotionAnalysisCalculator::SubtractMetaMotion(
    const CameraMotion& meta_motion, RegionFlowFeatureList* features) {
  if (meta_motion.mixture_homography().model_size() > 0) {
    CHECK(row_weights_ != nullptr);
    RegionFlowFeatureListViaTransform(meta_motion.mixture_homography(),
                                      features, -1.0f,
                                      1.0f,  // subtract transformed.
                                      true,  // replace feature loc.
                                      row_weights_.get());
  } else {
    RegionFlowFeatureListViaTransform(meta_motion.homography(), features, -1.0f,
                                      1.0f,   // subtract transformed.
                                      true);  // replace feature loc.
  }

  // Clamp transformed features to domain and handle outliers.
  const float domain_diam =
      hypot(features->frame_width(), features->frame_height());
  const float motion_mag = meta_motion.average_magnitude();
  // Same irls fraction as used by MODEL_MIXTURE_HOMOGRAPHY scaling in
  // MotionEstimation.
  const float irls_fraction = options_.analysis_options()
                                  .motion_options()
                                  .irls_mixture_fraction_scale() *
                              options_.analysis_options()
                                  .motion_options()
                                  .irls_motion_magnitude_fraction();
  float err_scale = std::max(1.0f, motion_mag * irls_fraction);

  const float max_err =
      options_.meta_outlier_domain_ratio() * domain_diam * err_scale;
  const float max_err_sq = max_err * max_err;

  for (auto& feature : *features->mutable_feature()) {
    feature.set_x(
        std::max(0.0f, std::min(features->frame_width() - 1.0f, feature.x())));
    feature.set_y(
        std::max(0.0f, std::min(features->frame_height() - 1.0f, feature.y())));
    // Label anything with large residual motion an outlier.
    if (FeatureFlow(feature).Norm2() > max_err_sq) {
      feature.set_irls_weight(0.0f);
    }
  }
}

void MotionAnalysisCalculator::AddMetaMotion(
    const CameraMotion& meta_motion, const RegionFlowFeatureList& meta_features,
    RegionFlowFeatureList* features, CameraMotion* motion) {
  // Restore old feature location.
  CHECK_EQ(meta_features.feature_size(), features->feature_size());
  for (int k = 0; k < meta_features.feature_size(); ++k) {
    auto feature = features->mutable_feature(k);
    const auto& meta_feature = meta_features.feature(k);
    feature->set_x(meta_feature.x());
    feature->set_y(meta_feature.y());
    feature->set_dx(meta_feature.dx());
    feature->set_dy(meta_feature.dy());
  }

  // Composite camera motion.
  *motion = ComposeCameraMotion(*motion, meta_motion);
  // Restore type from metadata, i.e. do not declare motions as invalid.
  motion->set_type(meta_motion.type());
  motion->set_match_frame(-1);
}

void MotionAnalysisCalculator::AppendCameraMotionsFromHomographies(
    const std::deque<Homography>& homographies, bool append_identity,
    std::deque<CameraMotion>* camera_motions,
    std::deque<RegionFlowFeatureList>* features) {
  CHECK(camera_motions);
  CHECK(features);

  CameraMotion identity;
  identity.set_frame_width(frame_width_);
  identity.set_frame_height(frame_height_);

  *identity.mutable_translation() = TranslationModel();
  *identity.mutable_linear_similarity() = LinearSimilarityModel();
  *identity.mutable_homography() = Homography();
  identity.set_type(CameraMotion::VALID);
  identity.set_match_frame(0);

  RegionFlowFeatureList empty_list;
  empty_list.set_long_tracks(true);
  empty_list.set_match_frame(-1);
  empty_list.set_frame_width(frame_width_);
  empty_list.set_frame_height(frame_height_);

  if (append_identity) {
    camera_motions->push_back(identity);
    features->push_back(empty_list);
  }

  const int models_per_frame = options_.meta_models_per_frame();
  CHECK_GT(models_per_frame, 0) << "At least one model per frame is needed";
  CHECK_EQ(0, homographies.size() % models_per_frame);
  const int num_frames = homographies.size() / models_per_frame;

  // Heuristic sigma, similar to what we use for rolling shutter removal.
  const float mixture_sigma = 1.0f / models_per_frame;

  if (row_weights_ == nullptr) {
    row_weights_.reset(new MixtureRowWeights(frame_height_,
                                             frame_height_ / 10,  // 10% margin
                                             mixture_sigma * frame_height_,
                                             1.0f, models_per_frame));
  }

  for (int f = 0; f < num_frames; ++f) {
    MixtureHomography mix_homog;
    const int model_start = f * models_per_frame;

    for (int k = 0; k < models_per_frame; ++k) {
      const Homography& homog = homographies[model_start + k];
      *mix_homog.add_model() = ModelInvert(homog);
    }

    CameraMotion c = identity;
    c.set_match_frame(-1);

    if (mix_homog.model_size() > 1) {
      *c.mutable_mixture_homography() = mix_homog;
      c.set_mixture_row_sigma(mixture_sigma);

      for (int k = 0; k < models_per_frame; ++k) {
        c.add_mixture_inlier_coverage(1.0f);
      }
      *c.add_mixture_homography_spectrum() = mix_homog;
      c.set_rolling_shutter_motion_index(0);

      *c.mutable_homography() = ProjectViaFit<Homography>(
          mix_homog, frame_width_, frame_height_, row_weights_.get());
    } else {
      // Guaranteed to exist because to check that models_per_frame > 0 above.
      *c.mutable_homography() = mix_homog.model(0);
    }

    // Project remaining motions down.
    *c.mutable_linear_similarity() = ProjectViaFit<LinearSimilarityModel>(
        c.homography(), frame_width_, frame_height_);
    *c.mutable_translation() = ProjectViaFit<TranslationModel>(
        c.homography(), frame_width_, frame_height_);

    c.set_average_magnitude(
        std::hypot(c.translation().dx(), c.translation().dy()));

    camera_motions->push_back(c);
    features->push_back(empty_list);
  }
}

}  // namespace mediapipe
