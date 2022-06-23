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

#include "mediapipe/util/sequence/media_sequence.h"

#include <cmath>
#include <limits>

#include "absl/strings/str_split.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/sequence/media_sequence_util.h"

namespace mediapipe {
namespace mediasequence {

namespace {

// Decodes the image header to get metadata as strings and ints.
bool ImageMetadata(const std::string& image_str, std::string* format_string,
                   int* width, int* height, int* channels) {
  // Determine the image encoding by matching known header bytes.
  if (image_str[0] == static_cast<char>(0x89) && image_str[1] == 'P' &&
      image_str[2] == 'N' && image_str[3] == 'G') {
    *format_string = "PNG";
  } else if (image_str[0] == static_cast<char>(0xFF) &&
             image_str[1] == static_cast<char>(0xD8) &&
             image_str[image_str.size() - 2] == static_cast<char>(0xFF) &&
             image_str[image_str.size() - 1] == static_cast<char>(0xD9)) {
    *format_string = "JPEG";
  } else {
    *format_string = "UNKNOWN";
  }
  auto buf = reinterpret_cast<void*>(const_cast<char*>(image_str.data()));
  cv::Mat img = cv::imdecode(cv::Mat(/*rows=*/image_str.size(),
                                     /*cols=*/1, CV_8UC1, buf),
                             -1 /*cv::ImreadModes::IMREAD_UNCHANGED*/);
  if (img.data == nullptr) {
    return false;
  }
  *width = img.cols;
  *height = img.rows;
  *channels = img.channels();
  return true;
}

// Finds the nearest timestamp in a FeatureList of timestamps. The FeatureList
// must contain int64 values and only the first value at each step is used.
int NearestIndex(int64 timestamp,
                 const tensorflow::FeatureList& int64_feature_list) {
  int64 closest_distance = std::numeric_limits<int64>::max();
  int index = -1;
  for (int i = 0; i < int64_feature_list.feature_size(); ++i) {
    int64 current_value = int64_feature_list.feature(i).int64_list().value(0);
    int64 current_distance = std::abs(current_value - timestamp);
    if (current_distance < closest_distance) {
      index = i;
      closest_distance = current_distance;
    }
  }
  return index;
}

// Find the numerical sampling rate between two values in seconds if the input
// timestamps are in microseconds.
float TimestampsToRate(int64 first_timestamp, int64 second_timestamp) {
  int64 timestamp_diff = second_timestamp - first_timestamp;
  // convert from microseconds to seconds.
  float rate = 1.0 / (static_cast<float>(timestamp_diff) / 1000000);
  return rate;
}

// Sets the values of "clip/number_of_frames", "clip/keyframe/index",
// "segment/start/index" and "segment/end/index" by finding the closest
// timestamps in the "image/timestamp" FeatureList if image timestamps are
// present.
absl::Status ReconcileAnnotationIndicesByImageTimestamps(
    tensorflow::SequenceExample* sequence) {
  if (GetImageTimestampSize(*sequence) == 0) {
    return absl::OkStatus();
  }
  int index;

  // clip/segment/index
  if (HasSegmentStartTimestamp(*sequence)) {
    int segment_size = GetSegmentStartTimestampSize(*sequence);
    RET_CHECK_EQ(GetSegmentEndTimestampSize(*sequence), segment_size)
        << "Expected an equal number of segment timestamps, but found "
        << "start: " << segment_size
        << ", end: " << GetSegmentEndTimestampSize(*sequence);

    std::vector<int64> start_indices;
    start_indices.reserve(segment_size);
    for (const int64& timestamp : GetSegmentStartTimestamp(*sequence)) {
      index = NearestIndex(timestamp,
                           GetFeatureList(*sequence, kImageTimestampKey));
      start_indices.push_back(index);
    }
    SetSegmentStartIndex(start_indices, sequence);

    std::vector<int64> end_indices;
    end_indices.reserve(segment_size);
    for (const int64& timestamp : GetSegmentEndTimestamp(*sequence)) {
      index = NearestIndex(timestamp,
                           GetFeatureList(*sequence, kImageTimestampKey));
      end_indices.push_back(index);
    }
    SetSegmentEndIndex(end_indices, sequence);
  }
  return absl::OkStatus();
}

// Sets the values of "image/format", "image/channels", "image/height",
// "image/width", and "image/frame_rate" based image metadata and timestamps.
absl::Status ReconcileMetadataImages(const std::string& prefix,
                                     tensorflow::SequenceExample* sequence) {
  if (GetImageEncodedSize(prefix, *sequence) == 0) {
    return absl::OkStatus();
  }
  std::string format;
  int height, width, channels;
  RET_CHECK(ImageMetadata(GetImageEncodedAt(prefix, *sequence, 0), &format,
                          &width, &height, &channels))
      << "Failure to decode image metadata of image: "
      << GetImageEncodedAt(prefix, *sequence, 0);
  SetImageFormat(prefix, format, sequence);
  SetImageHeight(prefix, height, sequence);
  SetImageWidth(prefix, width, sequence);
  SetImageChannels(prefix, channels, sequence);

  if (GetImageTimestampSize(prefix, *sequence) > 1) {
    float rate = TimestampsToRate(GetImageTimestampAt(prefix, *sequence, 0),
                                  GetImageTimestampAt(prefix, *sequence, 1));
    SetImageFrameRate(prefix, rate, sequence);
  }
  return absl::OkStatus();
}

// Sets the values of "feature/${TAG}/dimensions", and
// "feature/${TAG}/frame_rate" for each float list feature TAG. If the
// dimensions are already present as a context feature, this method verifies
// the number of elements in the feature. Otherwise, it will write the
// dimensions as a 1D vector with the number of elements.
absl::Status ReconcileMetadataFeatureFloats(
    tensorflow::SequenceExample* sequence) {
  // Loop through all keys and see if they contain "/feature/floats"
  // If so, check dimensions and set rate.
  for (const auto& key_value : sequence->feature_lists().feature_list()) {
    const std::string& key = key_value.first;
    if (absl::StrContains(key, kFeatureFloatsKey)) {
      const auto prefix = key.substr(0, key.find(kFeatureFloatsKey) - 1);
      int number_of_elements = GetFeatureFloatsAt(prefix, *sequence, 0).size();
      if (HasFeatureDimensions(prefix, *sequence) &&
          !GetFeatureDimensions(prefix, *sequence).empty()) {
        int64 product = 1;
        for (int64 value : GetFeatureDimensions(prefix, *sequence)) {
          product *= value;
        }
        RET_CHECK_EQ(number_of_elements, product)
            << "The number of elements in float feature_list " << prefix
            << "/feature/floats does not match the dimensions: "
            << number_of_elements;
      } else {
        SetFeatureDimensions(prefix, {number_of_elements}, sequence);
      }

      if (GetFeatureTimestampSize(prefix, *sequence) > 1) {
        float rate =
            TimestampsToRate(GetFeatureTimestampAt(prefix, *sequence, 0),
                             GetFeatureTimestampAt(prefix, *sequence, 1));
        SetFeatureRate(prefix, rate, sequence);
      }
    }
  }
  return absl::OkStatus();
}

// Go through all bounding box annotations and move the annotation to the
// nearest image frame with a timestamp. If timestamps are not present, does
// nothing. If two or more annotations are closest to the same frame, then only
// the closest annotation is saved. This matches the behavior of downsampling
// images streams in time.
absl::Status ReconcileMetadataBoxAnnotations(
    const std::string& prefix, tensorflow::SequenceExample* sequence) {
  int num_bboxes = GetBBoxTimestampSize(prefix, *sequence);
  int num_frames = GetImageTimestampSize(*sequence);
  if (num_bboxes && num_frames) {
    // If no one has indicated which frames are annotated, assume annotations
    // are dense.
    if (GetBBoxIsAnnotatedSize(prefix, *sequence) == 0) {
      for (int i = 0; i < num_bboxes; ++i) {
        AddBBoxIsAnnotated(prefix, true, sequence);
      }
    }
    RET_CHECK_EQ(num_bboxes, GetBBoxIsAnnotatedSize(prefix, *sequence))
        << "Expected number of BBox timestamps and annotation marks to match.";
    // Update num_bboxes.
    if (GetBBoxSize(prefix, *sequence) > 0) {
      std::string xmin_key = merge_prefix(prefix, kRegionBBoxXMinKey);
      auto* bbox_feature_list = MutableFeatureList(xmin_key, sequence);
      RET_CHECK_EQ(num_bboxes, bbox_feature_list->feature_size())
          << "Expected number of BBox timestamps and boxes to match.";
      ClearBBoxNumRegions(prefix, sequence);
      for (int i = 0; i < num_bboxes; ++i) {
        AddBBoxNumRegions(
            prefix, bbox_feature_list->feature(i).float_list().value_size(),
            sequence);
      }
    }
    if (GetPointSize(prefix, *sequence) > 0) {
      std::string x_key = merge_prefix(prefix, kRegionPointXKey);
      auto* region_feature_list = MutableFeatureList(x_key, sequence);
      RET_CHECK_EQ(num_bboxes, region_feature_list->feature_size())
          << "Expected number of BBox timestamps and boxes to match.";
      ClearBBoxNumRegions(prefix, sequence);
      for (int i = 0; i < num_bboxes; ++i) {
        AddBBoxNumRegions(
            prefix, region_feature_list->feature(i).float_list().value_size(),
            sequence);
      }
    }
    if (Get3dPointSize(prefix, *sequence) > 0) {
      std::string x_key = merge_prefix(prefix, kRegion3dPointXKey);
      auto* region_feature_list = MutableFeatureList(x_key, sequence);
      RET_CHECK_EQ(num_bboxes, region_feature_list->feature_size())
          << "Expected number of BBox timestamps and boxes to match.";
      ClearBBoxNumRegions(prefix, sequence);
      for (int i = 0; i < num_bboxes; ++i) {
        AddBBoxNumRegions(
            prefix, region_feature_list->feature(i).float_list().value_size(),
            sequence);
      }
    }
    // Collect which timestamps currently match to which indices in timestamps.
    // skip empty timestamps.
    // Requires sorted indices.
    ::std::vector<int64> box_timestamps(num_bboxes);
    int bbox_index = 0;
    std::string timestamp_key = merge_prefix(prefix, kRegionTimestampKey);
    for (auto& feature : GetFeatureList(*sequence, timestamp_key).feature()) {
      box_timestamps[bbox_index] = feature.int64_list().value(0);
      ++bbox_index;
    }
    ::std::vector<int32> box_is_annotated(num_bboxes);
    bbox_index = 0;
    std::string is_annotated_key = merge_prefix(prefix, kRegionIsAnnotatedKey);
    for (auto& feature :
         GetFeatureList(*sequence, is_annotated_key).feature()) {
      box_is_annotated[bbox_index] = feature.int64_list().value(0);
      ++bbox_index;
    }
    ::std::vector<int64> image_timestamps(num_frames);
    int frame_index = 0;
    for (auto& feature :
         GetFeatureList(*sequence, kImageTimestampKey).feature()) {
      image_timestamps[frame_index] = feature.int64_list().value(0);
      ++frame_index;
    }
    // Collect which bbox timestamps are closest to which image indices.
    ::std::vector<int> bbox_index_if_annotated(num_frames, -1);
    int box_index = 0;
    int image_index = 0;
    while (box_index < num_bboxes) {
      // leave unannotated boxes at -1.
      if (!box_is_annotated[box_index]) {
        box_index += 1;
        // annotated boxes should updated their closest index.
      } else if (image_index >= num_frames - 1 ||
                 llabs(image_timestamps[image_index] -
                       box_timestamps[box_index]) <
                     llabs(image_timestamps[image_index + 1] -
                           box_timestamps[box_index])) {
        // Only overwrite with a new value if no value is present or this is
        // closer in time.
        if (bbox_index_if_annotated[image_index] == -1 ||
            llabs(image_timestamps[image_index] -
                  box_timestamps[bbox_index_if_annotated[image_index]]) >
                llabs(image_timestamps[image_index] -
                      box_timestamps[box_index])) {
          bbox_index_if_annotated[image_index] = box_index;
        }
        box_index += 1;
      } else {
        image_index += 1;
      }
    }
    // Only update unmodified bbox timestamp if it doesn't exist to prevent
    // overwriting with modified values.
    if (!GetUnmodifiedBBoxTimestampSize(prefix, *sequence)) {
      for (int i = 0; i < num_frames; ++i) {
        const int bbox_index = bbox_index_if_annotated[i];
        if (bbox_index >= 0 &&
            GetBBoxIsAnnotatedAt(prefix, *sequence, bbox_index)) {
          AddUnmodifiedBBoxTimestamp(prefix, box_timestamps[bbox_index],
                                     sequence);
        }
      }
    }
    // store some new feature_lists in a temporary sequence
    std::string expected_prefix = merge_prefix(prefix, "region/");
    ::tensorflow::SequenceExample tmp_seq;
    for (const auto& key_value : sequence->feature_lists().feature_list()) {
      const std::string& key = key_value.first;
      if (::absl::StartsWith(key, expected_prefix)) {
        // create a new set of values and swap them in.
        tmp_seq.Clear();
        auto* old_feature_list = MutableFeatureList(key, sequence);
        auto* new_feature_list = MutableFeatureList(key, &tmp_seq);
        if (key != merge_prefix(prefix, kUnmodifiedRegionTimestampKey)) {
          RET_CHECK_EQ(num_bboxes, old_feature_list->feature().size())
              << "Expected number of BBox timestamps to match number of "
                 "entries "
              << "in " << key;
          for (int i = 0; i < num_frames; ++i) {
            if (bbox_index_if_annotated[i] >= 0) {
              if (key == merge_prefix(prefix, kRegionTimestampKey)) {
                new_feature_list->add_feature()
                    ->mutable_int64_list()
                    ->add_value(image_timestamps[i]);
              } else {
                *new_feature_list->add_feature() =
                    old_feature_list->feature(bbox_index_if_annotated[i]);
              }
            } else {
              // Add either a default value or an empty.
              if (key == merge_prefix(prefix, kRegionIsAnnotatedKey)) {
                new_feature_list->add_feature()
                    ->mutable_int64_list()
                    ->add_value(0);
              } else if (key == merge_prefix(prefix, kRegionNumRegionsKey)) {
                new_feature_list->add_feature()
                    ->mutable_int64_list()
                    ->add_value(0);
              } else if (key == merge_prefix(prefix, kRegionTimestampKey)) {
                new_feature_list->add_feature()
                    ->mutable_int64_list()
                    ->add_value(image_timestamps[i]);
              } else {
                new_feature_list->add_feature();  // Adds an empty.
              }
            }
          }
          *old_feature_list = *new_feature_list;
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ReconcileMetadataRegionAnnotations(
    tensorflow::SequenceExample* sequence) {
  // Copy keys for fixed iteration order while updating feature_lists.
  std::vector<std::string> keys;
  for (const auto& key_value : sequence->feature_lists().feature_list()) {
    keys.push_back(key_value.first);
  }
  for (const std::string& key : keys) {
    if (::absl::StrContains(key, kRegionTimestampKey)) {
      std::string prefix = "";
      if (key != kRegionTimestampKey) {
        prefix = key.substr(0, key.size() - sizeof(kRegionTimestampKey));
      }
      RET_CHECK_OK(ReconcileMetadataBoxAnnotations(prefix, sequence));
    }
  }
  return absl::OkStatus();
}
}  // namespace

int GetBBoxSize(const std::string& prefix,
                const tensorflow::SequenceExample& sequence) {
  return GetBBoxXMinSize(prefix, sequence);
}

std::vector<::mediapipe::Location> GetBBoxAt(
    const std::string& prefix, const tensorflow::SequenceExample& sequence,
    int index) {
  std::vector<::mediapipe::Location> bboxes;
  const auto& xmins = GetBBoxXMinAt(prefix, sequence, index);
  const auto& ymins = GetBBoxYMinAt(prefix, sequence, index);
  const auto& xmaxs = GetBBoxXMaxAt(prefix, sequence, index);
  const auto& ymaxs = GetBBoxYMaxAt(prefix, sequence, index);
  bboxes.reserve(xmins.size());
  for (int i = 0; i < xmins.size(); ++i) {
    bboxes.push_back(::mediapipe::Location::CreateRelativeBBoxLocation(
        xmins[i], ymins[i], xmaxs[i] - xmins[i], ymaxs[i] - ymins[i]));
  }
  return bboxes;
}

void AddBBox(const std::string& prefix,
             const std::vector<::mediapipe::Location>& bboxes,
             tensorflow::SequenceExample* sequence) {
  ::std::vector<float> xmins;
  ::std::vector<float> ymins;
  ::std::vector<float> xmaxs;
  ::std::vector<float> ymaxs;
  for (auto& bbox : bboxes) {
    const auto& rect = bbox.GetRelativeBBox();
    xmins.push_back(rect.xmin());
    ymins.push_back(rect.ymin());
    xmaxs.push_back(rect.xmax());
    ymaxs.push_back(rect.ymax());
  }
  AddBBoxXMin(prefix, xmins, sequence);
  AddBBoxYMin(prefix, ymins, sequence);
  AddBBoxXMax(prefix, xmaxs, sequence);
  AddBBoxYMax(prefix, ymaxs, sequence);
}

void ClearBBox(const std::string& prefix,
               tensorflow::SequenceExample* sequence) {
  ClearBBoxXMin(prefix, sequence);
  ClearBBoxYMin(prefix, sequence);
  ClearBBoxXMax(prefix, sequence);
  ClearBBoxYMax(prefix, sequence);
}

int GetPointSize(const std::string& prefix,
                 const tensorflow::SequenceExample& sequence) {
  return GetBBoxPointXSize(prefix, sequence);
}

std::vector<::std::pair<float, float>> GetPointAt(
    const std::string& prefix, const tensorflow::SequenceExample& sequence,
    int index) {
  const auto& ys = GetBBoxPointYAt(prefix, sequence, index);
  const auto& xs = GetBBoxPointXAt(prefix, sequence, index);
  std::vector<::std::pair<float, float>> points(ys.size());
  for (int i = 0; i < xs.size(); ++i) {
    points[i].first = ys[i];
    points[i].second = xs[i];
  }
  return points;
}

void AddPoint(const std::string& prefix,
              const std::vector<::std::pair<float, float>>& points,
              tensorflow::SequenceExample* sequence) {
  ::std::vector<float> xs;
  ::std::vector<float> ys;
  for (auto& point : points) {
    ys.push_back(point.first);
    xs.push_back(point.second);
  }
  AddBBoxPointY(prefix, ys, sequence);
  AddBBoxPointX(prefix, xs, sequence);
}

void ClearPoint(const std::string& prefix,
                tensorflow::SequenceExample* sequence) {
  ClearBBoxPointY(prefix, sequence);
  ClearBBoxPointX(prefix, sequence);
}

int Get3dPointSize(const std::string& prefix,
                   const tensorflow::SequenceExample& sequence) {
  return GetBBox3dPointXSize(prefix, sequence);
}

std::vector<::std::tuple<float, float, float>> Get3dPointAt(
    const std::string& prefix, const tensorflow::SequenceExample& sequence,
    int index) {
  const auto& xs = GetBBox3dPointXAt(prefix, sequence, index);
  const auto& ys = GetBBox3dPointYAt(prefix, sequence, index);
  const auto& zs = GetBBox3dPointZAt(prefix, sequence, index);
  std::vector<::std::tuple<float, float, float>> points(ys.size());
  for (int i = 0; i < xs.size(); ++i) {
    points[i] = std::make_tuple(xs[i], ys[i], zs[i]);
  }
  return points;
}

void Add3dPoint(const std::string& prefix,
                const std::vector<::std::tuple<float, float, float>>& points,
                tensorflow::SequenceExample* sequence) {
  ::std::vector<float> xs;
  ::std::vector<float> ys;
  ::std::vector<float> zs;
  for (auto& point : points) {
    xs.push_back(std::get<0>(point));
    ys.push_back(std::get<1>(point));
    zs.push_back(std::get<2>(point));
  }
  AddBBox3dPointX(prefix, xs, sequence);
  AddBBox3dPointY(prefix, ys, sequence);
  AddBBox3dPointZ(prefix, zs, sequence);
}

void Clear3dPoint(const std::string& prefix,
                  tensorflow::SequenceExample* sequence) {
  ClearBBox3dPointX(prefix, sequence);
  ClearBBox3dPointY(prefix, sequence);
  ClearBBox3dPointZ(prefix, sequence);
}

std::unique_ptr<mediapipe::Matrix> GetAudioFromFeatureAt(
    const std::string& prefix, const tensorflow::SequenceExample& sequence,
    int index) {
  const auto& flat_data = GetFeatureFloatsAt(prefix, sequence, index);
  CHECK(HasFeatureNumChannels(prefix, sequence))
      << "GetAudioAt requires num_channels context to be specified as key: "
      << merge_prefix(prefix, kFeatureNumChannelsKey);
  int num_channels = GetFeatureNumChannels(prefix, sequence);
  CHECK_EQ(flat_data.size() % num_channels, 0)
      << "The data size is not a multiple of the number of channels: "
      << flat_data.size() << " % " << num_channels << " = "
      << flat_data.size() % num_channels << " for sequence index " << index;
  auto output = absl::make_unique<mediapipe::Matrix>(
      num_channels, flat_data.size() / num_channels);
  std::copy(flat_data.begin(), flat_data.end(), output->data());
  return output;
}

void AddAudioAsFeature(const std::string& prefix,
                       const mediapipe::Matrix& audio,
                       tensorflow::SequenceExample* sequence) {
  auto* value_list =
      MutableFeatureList(merge_prefix(prefix, kFeatureFloatsKey), sequence)
          ->add_feature()
          ->mutable_float_list()
          ->mutable_value();
  mediapipe::proto_ns::RepeatedField<float>(
      audio.data(), audio.data() + audio.rows() * audio.cols())
      .Swap(value_list);
}

absl::Status ReconcileMetadata(bool reconcile_bbox_annotations,
                               bool reconcile_region_annotations,
                               tensorflow::SequenceExample* sequence) {
  RET_CHECK_OK(ReconcileAnnotationIndicesByImageTimestamps(sequence));
  RET_CHECK_OK(ReconcileMetadataImages("", sequence));
  RET_CHECK_OK(ReconcileMetadataImages(kForwardFlowPrefix, sequence));
  RET_CHECK_OK(ReconcileMetadataImages(kClassSegmentationPrefix, sequence));
  RET_CHECK_OK(ReconcileMetadataImages(kInstanceSegmentationPrefix, sequence));
  RET_CHECK_OK(ReconcileMetadataFeatureFloats(sequence));
  if (reconcile_bbox_annotations) {
    RET_CHECK_OK(ReconcileMetadataBoxAnnotations("", sequence));
  }
  if (reconcile_region_annotations) {
    RET_CHECK_OK(ReconcileMetadataRegionAnnotations(sequence));
  }
  // audio is always reconciled in the framework.
  return absl::OkStatus();
}

}  // namespace mediasequence
}  // namespace mediapipe
