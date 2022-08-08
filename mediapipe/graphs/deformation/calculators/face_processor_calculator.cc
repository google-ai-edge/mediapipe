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

#include <math.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>

#include <memory>
#include "Tensor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe
{
  namespace
  {

    constexpr char kImageSizeTag[] = "SIZE";
    constexpr char kVectorTag[] = "VECTOR";
    constexpr char kLandmarksTag[] = "LANDMARKS";
    constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
    constexpr char kSrcTensorTag[] = "SRC_TENSOR";
    constexpr char kDstTensorTag[] = "DST_TENSOR";

    inline bool HasImageTag(mediapipe::CalculatorContext *cc) { return false; }

    bool NormalizedtoPixelCoordinates(double normalized_x, double normalized_y, double normalized_z,
                                      int image_width, int image_height, double *x_px,
                                      double *y_px, double *z_px)
    {
      CHECK(x_px != nullptr);
      CHECK(y_px != nullptr);
      CHECK_GT(image_width, 0);
      CHECK_GT(image_height, 0);

      if (normalized_x < 0 || normalized_x > 1.0 || normalized_y < 0 ||
          normalized_y > 1.0 || normalized_z < 0 ||
          normalized_z > 1.0)
      {
        VLOG(1) << "Normalized coordinates must be between 0.0 and 1.0";
      }

      *x_px = static_cast<double>(normalized_x) * image_width;
      *y_px = static_cast<double>(normalized_y) * image_height;
      *z_px = static_cast<double>(normalized_z) * image_width;

      return true;
    }

    template <class LandmarkType>
    bool IsLandmarkVisibleAndPresent(const LandmarkType &landmark,
                                     bool utilize_visibility,
                                     float visibility_threshold,
                                     bool utilize_presence,
                                     float presence_threshold)
    {
      if (utilize_visibility && landmark.has_visibility() &&
          landmark.visibility() < visibility_threshold)
      {
        return false;
      }
      if (utilize_presence && landmark.has_presence() &&
          landmark.presence() < presence_threshold)
      {
        return false;
      }
      return true;
    }

  } // namespace

  class FaceProcessorCalculator : public CalculatorBase
  {
  public:
    FaceProcessorCalculator() = default;
    ~FaceProcessorCalculator() override = default;

    static absl::Status GetContract(CalculatorContract *cc);

    // From Calculator.
    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status RenderToCpu(
        CalculatorContext *cc);

    absl::Status SetData(CalculatorContext *cc);

    absl::Status ProcessImage(CalculatorContext *cc);

    static absl::StatusOr<string> ReadContentBlobFromFile(
        const string &unresolved_path)
    {
      ASSIGN_OR_RETURN(string resolved_path,
                       mediapipe::PathToResourceAsFile(unresolved_path),
                       _ << "Failed to resolve path! Path = " << unresolved_path);

      string content_blob;
      MP_RETURN_IF_ERROR(
          mediapipe::GetResourceContents(resolved_path, &content_blob))
          << "Failed to read content blob! Resolved path = " << resolved_path;

      return content_blob;
    }

    vector<string> index_names;
    map<string, vector<int>> indexes;

    map<string, Tensor<double>> masks;
    vector<vector<int>> _trianglesIndexes;
    Tensor<double> __facePts;
  };

  absl::Status FaceProcessorCalculator::GetContract(CalculatorContract *cc)
  {
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) ||
              cc->Inputs().HasTag(kNormLandmarksTag))
        << "None of the input streams are provided.";
    RET_CHECK(!(cc->Inputs().HasTag(kLandmarksTag) &&
                cc->Inputs().HasTag(kNormLandmarksTag)))
        << "Can only one type of landmark can be taken. Either absolute or "
           "normalized landmarks.";

    if (cc->Inputs().HasTag(kLandmarksTag))
    {
      cc->Inputs().Tag(kLandmarksTag).Set<vector<LandmarkList>>();
    }
    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      cc->Inputs().Tag(kNormLandmarksTag).Set<vector<NormalizedLandmarkList>>();
    }

    if (cc->Inputs().HasTag(kImageSizeTag))
    {
      cc->Inputs().Tag(kImageSizeTag).Set<pair<int, int>>();
    }
    if (cc->Outputs().HasTag(kSrcTensorTag))
    {
      cc->Outputs().Tag(kSrcTensorTag).Set<Tensor<double>>();
    }
    if (cc->Outputs().HasTag(kDstTensorTag))
    {
      cc->Outputs().Tag(kDstTensorTag).Set<Tensor<double>>();
    }

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::Process(CalculatorContext *cc)
  {
    MP_RETURN_IF_ERROR(SetData(cc));

    if (cc->Inputs().HasTag(kNormLandmarksTag) &&
        !cc->Inputs().Tag(kNormLandmarksTag).IsEmpty())
    {
      MP_RETURN_IF_ERROR(ProcessImage(cc));
    }

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::SetData(CalculatorContext *cc)
  {
    masks = {};
    _trianglesIndexes = {};

    string filename = "mediapipe/graphs/deformation/config/triangles.txt";
    string content_blob;
    ASSIGN_OR_RETURN(content_blob,
                     ReadContentBlobFromFile(filename),
                     _ << "Failed to read texture blob from file!");

    istringstream stream(content_blob);
    double points[854][3];
    vector<int> tmp;
    for (int i = 0; i < 854; ++i)
    {
      string line;
      tmp = {};
      for (int j = 0; j < 3; ++j)
      {
        stream >> points[i][j];
        tmp.push_back((int)points[i][j]);
      }
      _trianglesIndexes.push_back(tmp);
    }

    filename = "./mediapipe/graphs/deformation/config/index_names.txt";
    ASSIGN_OR_RETURN(content_blob,
                     ReadContentBlobFromFile(filename),
                     _ << "Failed to read texture blob from file!");
    istringstream stream2(content_blob);

    string line;
    vector<int> idxs;
    while (getline(stream2, line))
    {
      index_names.push_back(line);
    }
    stream2.clear();

    for (int i = 0; i < index_names.size(); i++)
    {
      filename = "./mediapipe/graphs/deformation/config/" + index_names[i] + ".txt";

      ASSIGN_OR_RETURN(content_blob,
                       ReadContentBlobFromFile(filename),
                       _ << "Failed to read texture blob from file!");
      stream2.str(content_blob);

      while (getline(stream2, line))
      {
        idxs.push_back(stoi(line));
      }
      indexes[index_names[i]] = idxs;

      idxs = {};
      stream2.clear();
    }

    double **zero_arr;
    for (int i = 0; i < index_names.size(); i++)
    {
      zero_arr = (double **)new double *[478];
      for (int j = 0; j < 478; j++)
      {
        zero_arr[j] = (double *)new double[1];
        zero_arr[j][0] = 0.0;
      }
      for (int j = 0; j < indexes[index_names[i]].size(); j++)
      {
        zero_arr[indexes[index_names[i]][j]][0] = 1;
      }
      masks[index_names[i]] = Tensor<double>(zero_arr, 478, 1);
    }

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::ProcessImage(CalculatorContext *cc)
  {
    double alfaNose = 1.2;
    double alfaLips = 0.4;
    double alfaCheekbones = 0.4;

    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
      const auto [image_width_, image_height_] = cc->Inputs().Tag(kImageSizeTag).Get<pair<int, int>>();

      const vector<NormalizedLandmarkList> &landmarks =
          cc->Inputs().Tag(kNormLandmarksTag).Get<vector<NormalizedLandmarkList>>();

      int n = 478;
      int m = 3;

      double **_points = (double **)new double *[n];
      for (int i = 0; i < n; i++)
        _points[i] = (double *)new double[m];

      for (int i = 0; i < landmarks[0].landmark_size(); ++i)
      {
        const NormalizedLandmark &landmark = landmarks[0].landmark(i);

        if (!IsLandmarkVisibleAndPresent<NormalizedLandmark>(
                landmark, false,
                0.0, false,
                0.0))
        {
          continue;
        }

        const auto &point = landmark;

        double x = -1;
        double y = -1;
        double z = -1;

        CHECK(NormalizedtoPixelCoordinates(point.x(), point.y(), point.z(), image_width_,
                                           image_height_, &x, &y, &z));
        _points[i][0] = x;
        _points[i][1] = y;
        _points[i][2] = z;
      }
      __facePts = Tensor<double>(_points, n, m);

      Tensor<double> ___facePts = __facePts - 0;

      Tensor<double> _X = __facePts.index(indexes["mediumNoseIndexes"]).index(Range::all(), Range(0, 1));
      Tensor<double> __YZ = __facePts.index(indexes["mediumNoseIndexes"]).index(Range::all(), Range(1, -1));
      Tensor<double> ___YZ = __YZ.concat(Tensor<double>(Mat::ones(9, 1, CV_64F)), 1);
      Tensor<double> _b = ___YZ.transpose().matmul(___YZ).inverse().matmul(___YZ.transpose()).matmul(_X);
      Tensor<double> _ort = Tensor<double>(Mat::ones(1, 1, CV_64F)).concat(-_b.index(Range(0, 2), Range::all()), 0);
      double _D = _b.at({2, 0}) / _ort.norm();
      _ort = _ort / _ort.norm();

      Tensor<double> _mask;
      Tensor<double> _dsts;
      vector<string> _indexes;
      _indexes = {"cheekbonesIndexes", "noseAllIndexes", "additionalNoseIndexes1", "additionalNoseIndexes2", "additionalNoseIndexes3"};

      vector<double> coeffs;
      coeffs = {alfaCheekbones * 0.2, alfaNose * 0.2, alfaNose * 0.1, alfaNose * 0.05, alfaNose * 0.025};

      _mask = masks["faceOvalIndexes"];
      _dsts = _mask * (___facePts.matmul(_ort) - _D);
      ___facePts = ___facePts + _dsts.matmul(_ort.transpose()) * 0.05;
      __facePts = __facePts + _dsts.matmul(_ort.transpose()) * 0.05;

      for (int i = 0; i < 5; i++)
      {
        _mask = masks[_indexes[i]];
        _dsts = _mask * (___facePts.matmul(_ort) - _D);
        ___facePts = ___facePts - coeffs[i] * _dsts.matmul(_ort.transpose());
      }

      _D = -1;
      Tensor<double> _lipsSupprotPoint = (___facePts.index(11) + ___facePts.index(16)) / 2;
      Tensor<double> _ABC = _lipsSupprotPoint.concat(___facePts.index(291), 0).concat(___facePts.index(61), 0).inverse().matmul(Tensor<double>(Mat::ones(3, 1, CV_64F))) * _D;
      _D = _D / _ABC.norm();
      _ort = _ABC / _ABC.norm();

      _indexes = {"upperLipCnt", "lowerLipCnt", "widerUpperLipPts1", "widerLowerLipPts1"};
      coeffs = {alfaLips, alfaLips * 0.5, alfaLips * 0.5, alfaLips * 0.25};

      for (int i = 0; i < 4; i++)
      {
        _mask = masks[_indexes[i]];
        _dsts = _mask * (___facePts.matmul(_ort) - _D);
        ___facePts = ___facePts + coeffs[i] * _dsts.matmul(_ort.transpose());
      }

      Tensor<double> tmp_order = ___facePts.index(_trianglesIndexes);
      tmp_order = -tmp_order.index(Range::all(), 2) - tmp_order.index(Range::all(), 5) - tmp_order.index(Range::all(), 8);
      tmp_order = tmp_order.transpose();
      vector<double> __order = tmp_order.get_1d_data();
      vector<int> _order = tmp_order.sort_indexes(__order);

      Tensor<double> _src = __facePts.index(_trianglesIndexes).index(_order);
      Tensor<double> _dst = ___facePts.index(_trianglesIndexes).index(_order);

      auto srcPtr = absl::make_unique<Tensor<double>>(_src);
      cc->Outputs().Tag(kSrcTensorTag).Add(srcPtr.release(), cc->InputTimestamp());

      auto dstPtr = absl::make_unique<Tensor<double>>(_dst);
      cc->Outputs().Tag(kDstTensorTag).Add(dstPtr.release(), cc->InputTimestamp());

      return absl::OkStatus();
    }
    else
    {
      return absl::OkStatus();
    }
  }

  REGISTER_CALCULATOR(FaceProcessorCalculator);
} // namespace mediapipe
