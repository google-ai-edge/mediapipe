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

#include <memory>

#include "exif.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_file_properties.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

namespace {

// 35 MM sensor has dimensions 36 mm x 24 mm, so diagonal length is
// sqrt(36^2 + 24^2).
static const double SENSOR_DIAGONAL_35MM = std::sqrt(1872.0);

absl::StatusOr<double> ComputeFocalLengthInPixels(int image_width,
                                                  int image_height,
                                                  double focal_length_35mm,
                                                  double focal_length_mm) {
  // TODO: Allow returning image file properties even when focal length
  // computation is not possible.
  if (image_width == 0 || image_height == 0) {
    return absl::InternalError(
        "Image dimensions should be non-zero to compute focal length in "
        "pixels.");
  }
  if (focal_length_mm == 0) {
    return absl::InternalError(
        "Focal length in mm should be non-zero to compute focal length in "
        "pixels.");
  }
  if (focal_length_35mm == 0) {
    return absl::InternalError(
        "Focal length in 35 mm should be non-zero to compute focal length in "
        "pixels.");
  }
  // Derived from
  // https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length#Calculation.
  /// Using focal_length_35mm = focal_length_mm * SENSOR_DIAGONAL_35MM /
  /// sensor_diagonal_mm, we can calculate the diagonal length of the sensor in
  /// millimeters i.e. sensor_diagonal_mm.
  double sensor_diagonal_mm =
      SENSOR_DIAGONAL_35MM / focal_length_35mm * focal_length_mm;
  // Note that for the following computations, the longer dimension is treated
  // as image width and the shorter dimension is treated as image height.
  int width = image_width;
  int height = image_height;
  if (image_height > image_width) {
    width = image_height;
    height = image_width;
  }
  double inv_aspect_ratio = (double)height / width;
  // Compute sensor width.
  /// Using Pythagoras theorem, sensor_width^2 + sensor_height^2 =
  /// sensor_diagonal_mm^2. We can substitute sensor_width / sensor_height with
  /// the aspect ratio calculated in pixels to compute the sensor width.
  double sensor_width = std::sqrt((sensor_diagonal_mm * sensor_diagonal_mm) /
                                  (1.0 + inv_aspect_ratio * inv_aspect_ratio));

  // Compute focal length in pixels.
  double focal_length_pixels = width * focal_length_mm / sensor_width;
  return focal_length_pixels;
}

absl::StatusOr<ImageFileProperties> GetImageFileProperties(
    const std::string& image_bytes) {
  easyexif::EXIFInfo result;
  int code = result.parseFrom(image_bytes);
  if (code) {
    return absl::InternalError("Error parsing EXIF, code: " +
                               std::to_string(code));
  }

  ImageFileProperties properties;
  properties.set_image_width(result.ImageWidth);
  properties.set_image_height(result.ImageHeight);
  properties.set_focal_length_mm(result.FocalLength);
  properties.set_focal_length_35mm(result.FocalLengthIn35mm);

  MP_ASSIGN_OR_RETURN(auto focal_length_pixels,
                      ComputeFocalLengthInPixels(properties.image_width(),
                                                 properties.image_height(),
                                                 properties.focal_length_35mm(),
                                                 properties.focal_length_mm()));
  properties.set_focal_length_pixels(focal_length_pixels);

  return properties;
}

}  // namespace

// Calculator to extract EXIF information from an image file. The input is
// a string containing raw byte data from a file, and the output is an
// ImageFileProperties proto object with the relevant fields filled in.
// The calculator accepts the input as a stream or a side packet, and can output
// the result as a stream or a side packet. The calculator checks that if an
// output stream is present, it outputs to that stream, and if not, it checks if
// it can output to a side packet.
//
// Example config with input and output streams:
// node {
//   calculator: "ImageFilePropertiesCalculator"
//   input_stream: "image_bytes"
//   output_stream: "image_properties"
// }
// Example config with input and output side packets:
// node {
//   calculator: "ImageFilePropertiesCalculator"
//   input_side_packet: "image_bytes"
//   output_side_packet: "image_properties"
// }
class ImageFilePropertiesCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    if (cc->Inputs().NumEntries() != 0) {
      RET_CHECK(cc->Inputs().NumEntries() == 1);
      cc->Inputs().Index(0).Set<std::string>();
    } else {
      RET_CHECK(cc->InputSidePackets().NumEntries() == 1);
      cc->InputSidePackets().Index(0).Set<std::string>();
    }
    if (cc->Outputs().NumEntries() != 0) {
      RET_CHECK(cc->Outputs().NumEntries() == 1);
      cc->Outputs().Index(0).Set<::mediapipe::ImageFileProperties>();
    } else {
      RET_CHECK(cc->OutputSidePackets().NumEntries() == 1);
      cc->OutputSidePackets().Index(0).Set<::mediapipe::ImageFileProperties>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    if (cc->InputSidePackets().NumEntries() == 1) {
      const std::string& image_bytes =
          cc->InputSidePackets().Index(0).Get<std::string>();
      MP_ASSIGN_OR_RETURN(properties_, GetImageFileProperties(image_bytes));
      read_properties_ = true;
    }

    if (read_properties_ && cc->OutputSidePackets().NumEntries() == 1) {
      cc->OutputSidePackets().Index(0).Set(
          MakePacket<ImageFileProperties>(properties_));
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().NumEntries() == 1) {
      if (cc->Inputs().Index(0).IsEmpty()) {
        return absl::OkStatus();
      }
      const std::string& image_bytes = cc->Inputs().Index(0).Get<std::string>();
      MP_ASSIGN_OR_RETURN(properties_, GetImageFileProperties(image_bytes));
      read_properties_ = true;
    }
    if (read_properties_) {
      if (cc->Outputs().NumEntries() == 1) {
        cc->Outputs().Index(0).AddPacket(
            MakePacket<ImageFileProperties>(properties_)
                .At(cc->InputTimestamp()));
      } else {
        cc->OutputSidePackets().Index(0).Set(
            MakePacket<ImageFileProperties>(properties_)
                .At(mediapipe::Timestamp::Unset()));
      }
    }

    return absl::OkStatus();
  }

 private:
  ImageFileProperties properties_;
  bool read_properties_ = false;
};
REGISTER_CALCULATOR(ImageFilePropertiesCalculator);

}  // namespace mediapipe
