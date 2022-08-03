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
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe
{
  namespace
  {

    constexpr char kImageFrameTag[] = "IMAGE";
    constexpr char kVectorTag[] = "VECTOR";
    constexpr char kLandmarksTag[] = "LANDMARKS";
    constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";

    tuple<int, int> _normalized_to_pixel_coordinates(float normalized_x,
                                                          float normalized_y, int image_width, int image_height)
    {
      // Converts normalized value pair to pixel coordinates
      int x_px = min<int>(floor(normalized_x * image_width), image_width - 1);
      int y_px = min<int>(floor(normalized_y * image_height), image_height - 1);

      return {x_px, y_px};
    };

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
      // 2280

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
    absl::Status CreateRenderTargetCpu(CalculatorContext *cc,
                                       unique_ptr<Mat> &image_mat,
                                       ImageFormat::Format *target_format);

    absl::Status RenderToCpu(
        CalculatorContext *cc, const ImageFormat::Format &target_format,
        uchar *data_image, unique_ptr<Mat> &image_mat);

    absl::Status SetData(CalculatorContext *cc);

    absl::Status ProcessImage(CalculatorContext *cc,
                              ImageFormat::Format &target_format);

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

    // Indicates if image frame is available as input.
    bool image_frame_available_ = false;

    unique_ptr<Mat> image_mat;
    vector<string> index_names;
    map<string, vector<int>> indexes;

    map<string, Tensor<double>> masks;
    vector<vector<int>> _trianglesIndexes;
    Tensor<double> __facePts;

    int image_width_;
    int image_height_;

    Mat mat_image_;
  };

  absl::Status FaceProcessorCalculator::GetContract(CalculatorContract *cc)
  {
    CHECK_GE(cc->Inputs().NumEntries(), 1);

    if (cc->Inputs().HasTag(kImageFrameTag))
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
      CHECK(cc->Outputs().HasTag(kImageFrameTag));
    }

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

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    if (cc->Inputs().HasTag(kImageFrameTag) || HasImageTag(cc))
    {
      image_frame_available_ = true;
    }

    // Set the output header based on the input header (if present).
    const char *tag = kImageFrameTag;
    if (image_frame_available_ && !cc->Inputs().Tag(tag).Header().IsEmpty())
    {
      const auto &input_header =
          cc->Inputs().Tag(tag).Header().Get<VideoHeader>();
      auto *output_video_header = new VideoHeader(input_header);
      cc->Outputs().Tag(tag).SetHeader(Adopt(output_video_header));
    }

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().HasTag(kImageFrameTag) &&
        cc->Inputs().Tag(kImageFrameTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    // Initialize render target, drawn with OpenCV.
    ImageFormat::Format target_format;

    MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));

    mat_image_ = *image_mat.get();
    image_width_ = image_mat->cols;
    image_height_ = image_mat->rows;
    
    MP_RETURN_IF_ERROR(SetData(cc));
    
    if (cc->Inputs().HasTag(kNormLandmarksTag) &&
        !cc->Inputs().Tag(kNormLandmarksTag).IsEmpty())
    {
      MP_RETURN_IF_ERROR(ProcessImage(cc, target_format));
    }

    uchar *image_mat_ptr = image_mat->data;
    MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr, image_mat));

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::Close(CalculatorContext *cc)
  {
    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::RenderToCpu(
      CalculatorContext *cc, const ImageFormat::Format &target_format,
      uchar *data_image, unique_ptr<Mat> &image_mat)
  {
 
    auto output_frame = absl::make_unique<ImageFrame>(
        target_format, mat_image_.cols, mat_image_.rows);

    output_frame->CopyPixelData(target_format, mat_image_.cols, mat_image_.rows, data_image,
                                ImageFrame::kDefaultAlignmentBoundary);

    if (cc->Outputs().HasTag(kImageFrameTag))
    {
      cc->Outputs()
          .Tag(kImageFrameTag)
          .Add(output_frame.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::CreateRenderTargetCpu(
      CalculatorContext *cc, unique_ptr<Mat> &image_mat,
      ImageFormat::Format *target_format)
  {
    if (image_frame_available_)
    {
      const auto &input_frame =
          cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();

      int target_mat_type;
      switch (input_frame.Format())
      {
      case ImageFormat::SRGBA:
        *target_format = ImageFormat::SRGBA;
        target_mat_type = CV_8UC4;
        break;
      case ImageFormat::SRGB:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      case ImageFormat::GRAY8:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      default:
        return absl::UnknownError("Unexpected image frame format.");
        break;
      }

      image_mat = absl::make_unique<Mat>(
          input_frame.Height(), input_frame.Width(), target_mat_type);

      auto input_mat = formats::MatView(&input_frame);

      if (input_frame.Format() == ImageFormat::GRAY8)
      {
        Mat rgb_mat;
        cvtColor(input_mat, rgb_mat, CV_GRAY2RGBA);
        rgb_mat.copyTo(*image_mat);
      }
      else
      {
        input_mat.copyTo(*image_mat);
      }
    }
    else
    {
      image_mat = absl::make_unique<Mat>(
          1920, 1280, CV_8UC4,
          Scalar::all(255.0));
      *target_format = ImageFormat::SRGBA;
    }

    return absl::OkStatus();
  }

  absl::Status FaceProcessorCalculator::SetData(CalculatorContext *cc)
  {
    masks.clear();
    _trianglesIndexes.clear();
   
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

  absl::Status FaceProcessorCalculator::ProcessImage(CalculatorContext *cc,
                                                     ImageFormat::Format &target_format)
  {
    double alfaNose = 0.7;
	  double alfaLips = 0.2;
	  double alfaCheekbones = 0.2;

    if (cc->Inputs().HasTag(kNormLandmarksTag))
    {
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
    }

    cvtColor(mat_image_, mat_image_, COLOR_BGRA2RGB);
    Mat clone_image = mat_image_.clone();

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

    Mat outImage = mat_image_.clone();

    for (int i = 0; i < 854; ++i)
    {
      if (i == 246)
      {
        int pointer = 0;
      }

      Tensor<double> __t1 = _src.index(vector<int>{i});
      Tensor<double> __t2 = _dst.index(vector<int>{i});

      vector<Point> t1;
      vector<Point> t2;

      for (int i = 0; i < 3; ++i)
      {
        t1.push_back(Point(
            (int)(__t1.at(vector<int>{0, 3 * i})),
            (int)(__t1.at(vector<int>{0, 3 * i + 1}))));
        t2.push_back(Point(
            (int)(__t2.at(vector<int>{0, 3 * i})),
            (int)(__t2.at(vector<int>{0, 3 * i + 1}))));
      }

      Rect r1 = boundingRect(t1);
      Rect r2 = boundingRect(t2);
      Point2f srcTri[3];
      Point2f dstTri[3];
      vector<Point> t1Rect;
      vector<Point> t2Rect;

      for (int i = 0; i < 3; ++i)
      {
        srcTri[i] = Point2f(t1[i].x - r1.x, t1[i].y - r1.y);
        dstTri[i] = Point2f(t2[i].x - r2.x, t2[i].y - r2.y);
        t1Rect.push_back(Point(t1[i].x - r1.x, t1[i].y - r1.y));
        t2Rect.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y));
      }

      Mat _dst;
      Mat mask = Mat::zeros(r2.height, r2.width, CV_8U);
      fillConvexPoly(mask, t2Rect, Scalar(1.0, 1.0, 1.0), 16, 0);
      
      if (r1.x + r1.width < clone_image.cols && r1.x >= 0 && r1.x + r1.width >= 0 && r1.y >= 0 && r1.y 
      < clone_image.rows && r1.y + r1.height < clone_image.rows)
      {
        Mat imgRect = mat_image_(Range(r1.y, r1.y + r1.height), Range(r1.x, r1.x + r1.width));
        Mat warpMat = getAffineTransform(srcTri, dstTri);
        warpAffine(imgRect, _dst, warpMat, mask.size());
        
        for (int i = r2.y; i < r2.y + r2.height; ++i)
        {
          for (int j = r2.x; j < r2.x + r2.width; ++j)
          {
            if ((int)mask.at<uchar>(i - r2.y, j - r2.x) > 0)
            {
              outImage.at<Vec3b>(i, j) = _dst.at<Vec3b>(i - r2.y, j - r2.x);
            }
          }
        }
      }
    }
    cvtColor(outImage, *image_mat, COLOR_RGB2BGRA);

    return absl::OkStatus();
  }

  REGISTER_CALCULATOR(FaceProcessorCalculator);
} // namespace mediapipe
