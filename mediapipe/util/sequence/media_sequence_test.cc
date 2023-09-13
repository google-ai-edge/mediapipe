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

#include <algorithm>
#include <string>
#include <vector>

#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace mediapipe {
namespace mediasequence {
namespace {

TEST(MediaSequenceTest, RoundTripDatasetName) {
  tensorflow::SequenceExample sequence;
  std::string name = "test";
  SetExampleDatasetName(name, &sequence);
  ASSERT_EQ(GetExampleDatasetName(sequence), name);
}

TEST(MediaSequenceTest, RoundTripDatasetFlagString) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> flags = {"test", "overall", "special"};
  SetExampleDatasetFlagString(flags, &sequence);
  ASSERT_THAT(GetExampleDatasetFlagString(sequence),
              testing::ElementsAreArray(flags));
}

TEST(MediaSequenceTest, RoundTripMediaId) {
  tensorflow::SequenceExample sequence;
  std::string id = "test";
  SetClipMediaId(id, &sequence);
  ASSERT_EQ(GetClipMediaId(sequence), id);
}

TEST(MediaSequenceTest, RoundTripDataPath) {
  tensorflow::SequenceExample sequence;
  std::string path = "test/here";
  SetClipDataPath(path, &sequence);
  ASSERT_EQ(GetClipDataPath(sequence), path);
}

TEST(MediaSequenceTest, RoundTripEncodedMediaBytes) {
  tensorflow::SequenceExample sequence;
  std::string data = "This is a test";
  SetClipEncodedMediaBytes(data, &sequence);
  ASSERT_EQ(GetClipEncodedMediaBytes(sequence), data);
}

TEST(MediaSequenceTest, RoundTripEncodedVideoStartTimestamp) {
  tensorflow::SequenceExample sequence;
  int64_t data = 47;
  SetClipEncodedMediaStartTimestamp(data, &sequence);
  ASSERT_EQ(GetClipEncodedMediaStartTimestamp(sequence), data);
}

TEST(MediaSequenceTest, RoundTripClipStartTimestamp) {
  tensorflow::SequenceExample sequence;
  int timestamp = 5;
  ASSERT_FALSE(HasClipStartTimestamp(sequence));
  SetClipStartTimestamp(timestamp, &sequence);
  ASSERT_EQ(GetClipStartTimestamp(sequence), timestamp);
  ASSERT_TRUE(HasClipStartTimestamp(sequence));
}

TEST(MediaSequenceTest, RoundTripClipEndTimestamp) {
  tensorflow::SequenceExample sequence;
  int timestamp = 5;
  ASSERT_FALSE(HasClipEndTimestamp(sequence));
  SetClipEndTimestamp(timestamp, &sequence);
  ASSERT_EQ(GetClipEndTimestamp(sequence), timestamp);
  ASSERT_TRUE(HasClipEndTimestamp(sequence));
}

TEST(MediaSequenceTest, RoundTripClipLabelIndex) {
  tensorflow::SequenceExample sequence;
  std::vector<int64_t> label = {5, 3};
  SetClipLabelIndex(label, &sequence);
  ASSERT_THAT(GetClipLabelIndex(sequence), testing::ElementsAreArray(label));
}

TEST(MediaSequenceTest, RoundTripClipLabelString) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> label = {"test", "again"};
  SetClipLabelString(label, &sequence);
  ASSERT_THAT(GetClipLabelString(sequence), testing::ElementsAreArray(label));
}

TEST(MediaSequenceTest, RoundTripFloatListFrameRate) {
  tensorflow::SequenceExample sequence;
  std::string key = "key";
  float frame_rate = 10.0f;
  SetFeatureRate(key, frame_rate, &sequence);
  ASSERT_FLOAT_EQ(GetFeatureRate(key, sequence), frame_rate);
}

TEST(MediaSequenceTest, RoundTripSegmentStartTimestamp) {
  tensorflow::SequenceExample sequence;
  EXPECT_FALSE(HasContext(sequence, kSegmentStartTimestampKey));
  SetSegmentStartTimestamp(::std::vector<int64_t>({123, 456}), &sequence);
  ASSERT_EQ(2, GetSegmentStartTimestampSize(sequence));
  ASSERT_THAT(GetSegmentStartTimestamp(sequence),
              testing::ElementsAreArray(::std::vector<int64_t>({123, 456})));
}

TEST(MediaSequenceTest, RoundTripSegmentEndTimestamp) {
  tensorflow::SequenceExample sequence;
  EXPECT_FALSE(HasContext(sequence, kSegmentEndTimestampKey));
  SetSegmentEndTimestamp(::std::vector<int64_t>({123, 456}), &sequence);
  ASSERT_EQ(2, GetSegmentEndTimestampSize(sequence));
  ASSERT_THAT(GetSegmentEndTimestamp(sequence),
              testing::ElementsAreArray(::std::vector<int64_t>({123, 456})));
}

TEST(MediaSequenceTest, RoundTripSegmentStartIndex) {
  tensorflow::SequenceExample sequence;
  EXPECT_FALSE(HasContext(sequence, kSegmentStartIndexKey));
  SetSegmentStartIndex(::std::vector<int64_t>({123, 456}), &sequence);
  ASSERT_EQ(2, GetSegmentStartIndexSize(sequence));
  ASSERT_THAT(GetSegmentStartIndex(sequence),
              testing::ElementsAreArray(::std::vector<int64_t>({123, 456})));
}

TEST(MediaSequenceTest, RoundTripSegmentEndIndex) {
  tensorflow::SequenceExample sequence;
  EXPECT_FALSE(HasContext(sequence, kSegmentEndIndexKey));
  SetSegmentEndIndex(::std::vector<int64_t>({123, 456}), &sequence);
  ASSERT_EQ(2, GetSegmentEndIndexSize(sequence));
  ASSERT_THAT(GetSegmentEndIndex(sequence),
              testing::ElementsAreArray(::std::vector<int64_t>({123, 456})));
}

TEST(MediaSequenceTest, RoundTripSegmentLabelIndex) {
  tensorflow::SequenceExample sequence;
  EXPECT_FALSE(HasContext(sequence, kSegmentLabelIndexKey));
  SetSegmentLabelIndex(::std::vector<int64_t>({5, 7}), &sequence);
  ASSERT_EQ(2, GetSegmentLabelIndexSize(sequence));
  ASSERT_THAT(GetSegmentLabelIndex(sequence),
              testing::ElementsAreArray(::std::vector<int64_t>({5, 7})));
}

TEST(MediaSequenceTest, RoundTripSegmentLabelString) {
  tensorflow::SequenceExample sequence;
  EXPECT_FALSE(HasContext(sequence, kSegmentLabelStringKey));
  SetSegmentLabelString(::std::vector<std::string>({"walk", "run"}), &sequence);
  ASSERT_EQ(2, GetSegmentLabelStringSize(sequence));
  ASSERT_THAT(
      GetSegmentLabelString(sequence),
      testing::ElementsAreArray(::std::vector<std::string>({"walk", "run"})));
}

TEST(MediaSequenceTest, RoundTripSegmentLabelConfidence) {
  tensorflow::SequenceExample sequence;
  EXPECT_FALSE(HasContext(sequence, kSegmentLabelConfidenceKey));
  SetSegmentLabelConfidence(::std::vector<float>({0.7f, 0.3f}), &sequence);
  ASSERT_EQ(2, GetSegmentLabelConfidenceSize(sequence));
  ASSERT_THAT(GetSegmentLabelConfidence(sequence),
              testing::ElementsAreArray(::std::vector<float>({0.7f, 0.3f})));
  ClearSegmentLabelConfidence(&sequence);
  EXPECT_EQ(0, GetSegmentLabelConfidenceSize(sequence));
}

TEST(MediaSequenceTest, RoundTripImageWidthHeight) {
  tensorflow::SequenceExample sequence;
  int64_t height = 2;
  int64_t width = 3;
  SetImageHeight(height, &sequence);
  ASSERT_EQ(GetImageHeight(sequence), height);
  SetImageWidth(width, &sequence);
  ASSERT_EQ(GetImageWidth(sequence), width);
}

TEST(MediaSequenceTest, RoundTripForwardFlowWidthHeight) {
  tensorflow::SequenceExample sequence;
  int64_t height = 2;
  int64_t width = 3;
  SetForwardFlowHeight(height, &sequence);
  ASSERT_EQ(GetForwardFlowHeight(sequence), height);
  SetForwardFlowWidth(width, &sequence);
  ASSERT_EQ(GetForwardFlowWidth(sequence), width);
}

TEST(MediaSequenceTest, RoundTripClassSegmentationWidthHeightFormat) {
  tensorflow::SequenceExample sequence;
  int64_t height = 2;
  int64_t width = 3;
  std::string format = "JPEG";
  SetClassSegmentationHeight(height, &sequence);
  EXPECT_EQ(GetClassSegmentationHeight(sequence), height);
  SetClassSegmentationWidth(width, &sequence);
  EXPECT_EQ(GetClassSegmentationWidth(sequence), width);
  SetClassSegmentationFormat(format, &sequence);
  EXPECT_EQ(GetClassSegmentationFormat(sequence), format);
}

TEST(MediaSequenceTest, RoundTripClassSegmentationLabelIndex) {
  tensorflow::SequenceExample sequence;
  std::vector<int64_t> classes = {5, 3};
  SetClassSegmentationClassLabelIndex(classes, &sequence);
  ASSERT_THAT(GetClassSegmentationClassLabelIndex(sequence),
              testing::ElementsAreArray({5, 3}));
  ClearClassSegmentationClassLabelIndex(&sequence);
  EXPECT_EQ(GetClassSegmentationClassLabelIndexSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripClassSegmentationLabelString) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> classes = {"5", "3"};
  SetClassSegmentationClassLabelString(classes, &sequence);
  ASSERT_THAT(GetClassSegmentationClassLabelString(sequence),
              testing::ElementsAreArray({"5", "3"}));
  ClearClassSegmentationClassLabelString(&sequence);
  EXPECT_EQ(GetClassSegmentationClassLabelStringSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripInstanceSegmentationWidthHeightFormat) {
  tensorflow::SequenceExample sequence;
  int64_t height = 2;
  int64_t width = 3;
  std::string format = "JPEG";
  SetInstanceSegmentationHeight(height, &sequence);
  EXPECT_EQ(GetInstanceSegmentationHeight(sequence), height);
  SetInstanceSegmentationWidth(width, &sequence);
  EXPECT_EQ(GetInstanceSegmentationWidth(sequence), width);
  SetInstanceSegmentationFormat(format, &sequence);
  EXPECT_EQ(GetInstanceSegmentationFormat(sequence), format);
}

TEST(MediaSequenceTest, RoundTripInstanceSegmentationClass) {
  tensorflow::SequenceExample sequence;
  std::vector<int64_t> classes = {5, 3};
  SetInstanceSegmentationObjectClassIndex(classes, &sequence);
  ASSERT_THAT(GetInstanceSegmentationObjectClassIndex(sequence),
              testing::ElementsAreArray({5, 3}));
  ClearInstanceSegmentationObjectClassIndex(&sequence);
  EXPECT_EQ(GetInstanceSegmentationObjectClassIndexSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripBBox) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7),
       Location::CreateRelativeBBoxLocation(0.3, 0.4, 0.2, 0.1)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.1, 0.2),
       Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.8)}};
  for (int i = 0; i < bboxes.size(); ++i) {
    AddBBox(bboxes[i], &sequence);
    ASSERT_EQ(GetBBoxSize(sequence), i + 1);
    const auto& sequence_bboxes = GetBBoxAt(sequence, i);
    for (int j = 0; j < sequence_bboxes.size(); ++j) {
      EXPECT_EQ(sequence_bboxes[j].GetRelativeBBox(),
                bboxes[i][j].GetRelativeBBox());
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxNumRegions) {
  tensorflow::SequenceExample sequence;
  std::vector<int> num_boxes = {5, 3};
  for (int i = 0; i < num_boxes.size(); ++i) {
    AddBBoxNumRegions(num_boxes[i], &sequence);
    ASSERT_EQ(GetBBoxNumRegionsSize(sequence), i + 1);
    ASSERT_EQ(GetBBoxNumRegionsAt(sequence, i), num_boxes[i]);
  }
  ClearBBoxNumRegions(&sequence);
  ASSERT_EQ(GetBBoxNumRegionsSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripBBoxLabelIndex) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<int64_t>> labels = {{5, 3}, {1, 2}};
  for (int i = 0; i < labels.size(); ++i) {
    AddBBoxLabelIndex(labels[i], &sequence);
    ASSERT_EQ(GetBBoxLabelIndexSize(sequence), i + 1);
    const auto& sequence_label = GetBBoxLabelIndexAt(sequence, i);
    for (int j = 0; j < sequence_label.size(); ++j) {
      ASSERT_EQ(sequence_label[j], labels[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxLabelString) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::string>> classes = {{"cat", "dog"}, {"dog"}};
  for (int i = 0; i < classes.size(); ++i) {
    AddBBoxLabelString(classes[i], &sequence);
    ASSERT_EQ(GetBBoxLabelStringSize(sequence), i + 1);
    const auto& sequence_classes = GetBBoxLabelStringAt(sequence, i);
    for (int j = 0; j < sequence_classes.size(); ++j) {
      ASSERT_EQ(sequence_classes[j], classes[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxClassIndex) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<int64_t>> classes = {{5, 3}, {1, 2}};
  for (int i = 0; i < classes.size(); ++i) {
    AddBBoxClassIndex(classes[i], &sequence);
    ASSERT_EQ(GetBBoxClassIndexSize(sequence), i + 1);
    const auto& sequence_classes = GetBBoxClassIndexAt(sequence, i);
    for (int j = 0; j < sequence_classes.size(); ++j) {
      ASSERT_EQ(sequence_classes[j], classes[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxClassString) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::string>> classes = {{"cat", "dog"}, {"dog"}};
  for (int i = 0; i < classes.size(); ++i) {
    AddBBoxClassString(classes[i], &sequence);
    ASSERT_EQ(GetBBoxClassStringSize(sequence), i + 1);
    const auto& sequence_classes = GetBBoxClassStringAt(sequence, i);
    for (int j = 0; j < sequence_classes.size(); ++j) {
      ASSERT_EQ(sequence_classes[j], classes[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxTrackIndex) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<int64_t>> tracks = {{5, 3}, {1, 2}};
  for (int i = 0; i < tracks.size(); ++i) {
    AddBBoxTrackIndex(tracks[i], &sequence);
    ASSERT_EQ(GetBBoxTrackIndexSize(sequence), i + 1);
    const auto& sequence_track = GetBBoxTrackIndexAt(sequence, i);
    for (int j = 0; j < sequence_track.size(); ++j) {
      ASSERT_EQ(sequence_track[j], tracks[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxTrackString) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::string>> tracks = {{"5", "3"}, {"1", "2"}};
  for (int i = 0; i < tracks.size(); ++i) {
    AddBBoxTrackString(tracks[i], &sequence);
    ASSERT_EQ(GetBBoxTrackStringSize(sequence), i + 1);
    const auto& sequence_track = GetBBoxTrackStringAt(sequence, i);
    for (int j = 0; j < sequence_track.size(); ++j) {
      ASSERT_EQ(sequence_track[j], tracks[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxTrackConfidence) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<float>> confidences = {{0.5, 0.3}, {0.1, 0.2}};
  for (int i = 0; i < confidences.size(); ++i) {
    AddBBoxTrackConfidence(confidences[i], &sequence);
    ASSERT_EQ(GetBBoxTrackConfidenceSize(sequence), i + 1);
    const auto& sequence_confidences = GetBBoxTrackConfidenceAt(sequence, i);
    for (int j = 0; j < sequence_confidences.size(); ++j) {
      ASSERT_EQ(sequence_confidences[j], confidences[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxTimestamp) {
  tensorflow::SequenceExample sequence;
  std::vector<int> timestamps = {5, 3};
  for (int i = 0; i < timestamps.size(); ++i) {
    AddBBoxTimestamp(timestamps[i], &sequence);
    ASSERT_EQ(GetBBoxTimestampSize(sequence), i + 1);
    ASSERT_EQ(GetBBoxTimestampAt(sequence, i), timestamps[i]);
  }
}

TEST(MediaSequenceTest, RoundTripUnmodifiedBBoxTimestamp) {
  tensorflow::SequenceExample sequence;
  std::vector<int> timestamps = {5, 3};
  for (int i = 0; i < timestamps.size(); ++i) {
    AddUnmodifiedBBoxTimestamp(timestamps[i], &sequence);
    ASSERT_EQ(GetUnmodifiedBBoxTimestampSize(sequence), i + 1);
    ASSERT_EQ(GetUnmodifiedBBoxTimestampAt(sequence, i), timestamps[i]);
  }
}

TEST(MediaSequenceTest, RoundTripBBoxIsAnnotated) {
  tensorflow::SequenceExample sequence;
  std::vector<int> is_annotated = {1, 0};
  for (int i = 0; i < is_annotated.size(); ++i) {
    AddBBoxIsAnnotated(is_annotated[i], &sequence);
    ASSERT_EQ(GetBBoxIsAnnotatedSize(sequence), i + 1);
    ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, i), is_annotated[i]);
  }
}

TEST(MediaSequenceTest, RoundTripBBoxEmbedding) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::string>> embeddings = {
      {"embedding00", "embedding01"}, {"embedding10", "embedding11"}};
  std::vector<std::vector<float>> confidences = {{0.7, 0.8}, {0.9, 0.95}};
  for (int i = 0; i < embeddings.size(); ++i) {
    AddBBoxEmbeddingEncoded("GT_KEY", embeddings[i], &sequence);
    ASSERT_EQ(GetBBoxEmbeddingEncodedSize("GT_KEY", sequence), i + 1);
    const auto& sequence_embeddings =
        GetBBoxEmbeddingEncodedAt("GT_KEY", sequence, i);
    EXPECT_THAT(sequence_embeddings, testing::ElementsAreArray(embeddings[i]));

    AddBBoxEmbeddingConfidence("GT_KEY", confidences[i], &sequence);
    ASSERT_EQ(GetBBoxEmbeddingConfidenceSize("GT_KEY", sequence), i + 1);
    const auto& sequence_confidences =
        GetBBoxEmbeddingConfidenceAt("GT_KEY", sequence, i);
    EXPECT_THAT(sequence_confidences,
                testing::ElementsAreArray(confidences[i]));
  }
}

TEST(MediaSequenceTest, RoundTripBBoxPoint) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::pair<float, float>>> points = {
      {{0.3, 0.5}, {0.4, 0.7}}, {{0.7, 0.5}, {0.3, 0.4}}};
  for (int i = 0; i < points.size(); ++i) {
    AddBBoxPoint(points[i], &sequence);
    ASSERT_EQ(GetBBoxPointSize(sequence), i + 1);
    const auto& sequence_points = GetBBoxPointAt(sequence, i);
    for (int j = 0; j < sequence_points.size(); ++j) {
      EXPECT_EQ(sequence_points[j], points[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBoxPointPrefixed) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::pair<float, float>>> points = {
      {{0.3, 0.5}, {0.4, 0.7}}, {{0.7, 0.5}, {0.3, 0.4}}};
  for (int i = 0; i < points.size(); ++i) {
    AddBBoxPoint("TEST", points[i], &sequence);
    ASSERT_EQ(GetBBoxPointSize("TEST", sequence), i + 1);
    const auto& sequence_points = GetBBoxPointAt("TEST", sequence, i);
    for (int j = 0; j < sequence_points.size(); ++j) {
      EXPECT_EQ(sequence_points[j], points[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripBBox3dPoint) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::tuple<float, float, float>>> points = {
      {std::make_tuple(0.3, 0.5, 0.1), std::make_tuple(0.4, 0.7, 0.2)},
      {std::make_tuple(0.7, 0.5, 0.3), std::make_tuple(0.3, 0.4, 0.4)}};
  for (int i = 0; i < points.size(); ++i) {
    AddBBox3dPoint(points[i], &sequence);
    ASSERT_EQ(GetBBox3dPointSize(sequence), i + 1);
    const auto& sequence_points = GetBBox3dPointAt(sequence, i);
    for (int j = 0; j < sequence_points.size(); ++j) {
      EXPECT_EQ(sequence_points[j], points[i][j]);
    }
  }
}

TEST(MediaSequenceTest, RoundTripRegionParts) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> parts = {"HEAD", "FEET"};
  SetBBoxParts(parts, &sequence);
  ASSERT_THAT(GetBBoxParts(sequence), testing::ElementsAreArray(parts));
  ClearBBoxParts(&sequence);
  EXPECT_EQ(GetBBoxPartsSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripPredictedBBox) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7),
       Location::CreateRelativeBBoxLocation(0.3, 0.4, 0.2, 0.1)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.1, 0.2),
       Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.8)}};
  for (int i = 0; i < bboxes.size(); ++i) {
    AddPredictedBBox(bboxes[i], &sequence);
    ASSERT_EQ(GetPredictedBBoxSize(sequence), i + 1);
    const auto& sequence_bboxes = GetPredictedBBoxAt(sequence, i);
    EXPECT_EQ(bboxes[i].size(), sequence_bboxes.size());
    for (int j = 0; j < sequence_bboxes.size(); ++j) {
      EXPECT_EQ(sequence_bboxes[j].GetRelativeBBox(),
                bboxes[i][j].GetRelativeBBox());
    }
  }
}

TEST(MediaSequenceTest, RoundTripPredictedBBoxTimestamp) {
  tensorflow::SequenceExample sequence;
  std::vector<int64_t> timestamps = {3, 6};
  for (int i = 0; i < timestamps.size(); ++i) {
    AddPredictedBBoxTimestamp(timestamps[i], &sequence);
    ASSERT_EQ(GetPredictedBBoxTimestampSize(sequence), i + 1);
    ASSERT_EQ(GetPredictedBBoxTimestampAt(sequence, i), timestamps[i]);
  }
}

TEST(MediaSequenceTest, RoundTripPredictedBBoxClasses) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::string>> classes = {{"cat", "dog"},
                                                   {"dog", "cat"}};
  for (int i = 0; i < classes.size(); ++i) {
    AddPredictedBBoxClassString(classes[i], &sequence);
    ASSERT_EQ(GetPredictedBBoxClassStringSize(sequence), i + 1);
    const auto& sequence_classes = GetPredictedBBoxClassStringAt(sequence, i);
    EXPECT_THAT(sequence_classes, testing::ElementsAreArray(classes[i]));
  }
}

TEST(MediaSequenceTest, RoundTripPredictedBBoxEmbedding) {
  tensorflow::SequenceExample sequence;
  std::vector<std::vector<std::string>> embeddings = {
      {"embedding00", "embedding01"}, {"embedding10", "embedding11"}};
  for (int i = 0; i < embeddings.size(); ++i) {
    AddBBoxEmbeddingEncoded("MY_KEY", embeddings[i], &sequence);
    ASSERT_EQ(GetBBoxEmbeddingEncodedSize("MY_KEY", sequence), i + 1);
    const auto& sequence_embeddings =
        GetBBoxEmbeddingEncodedAt("MY_KEY", sequence, i);
    EXPECT_THAT(sequence_embeddings, testing::ElementsAreArray(embeddings[i]));
  }
}

TEST(MediaSequenceTest, RoundTripImageEncoded) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> images = {"test", "again"};
  for (int i = 0; i < images.size(); ++i) {
    AddImageEncoded(images[i], &sequence);
    ASSERT_EQ(GetImageEncodedSize(sequence), i + 1);
    ASSERT_EQ(GetImageEncodedAt(sequence, i), images[i]);
  }
  ClearImageEncoded(&sequence);
  ASSERT_EQ(GetImageEncodedSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripClassSegmentationEncoded) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> images = {"test", "again"};
  for (int i = 0; i < images.size(); ++i) {
    AddClassSegmentationEncoded(images[i], &sequence);
    EXPECT_EQ(GetClassSegmentationEncodedSize(sequence), i + 1);
    EXPECT_EQ(GetClassSegmentationEncodedAt(sequence, i), images[i]);
  }
  ClearClassSegmentationEncoded(&sequence);
  EXPECT_EQ(GetClassSegmentationEncodedSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripInstanceSegmentationEncoded) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> images = {"test", "again"};
  for (int i = 0; i < images.size(); ++i) {
    AddInstanceSegmentationEncoded(images[i], &sequence);
    EXPECT_EQ(GetInstanceSegmentationEncodedSize(sequence), i + 1);
    EXPECT_EQ(GetInstanceSegmentationEncodedAt(sequence, i), images[i]);
  }
  ClearInstanceSegmentationEncoded(&sequence);
  EXPECT_EQ(GetInstanceSegmentationEncodedSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripSegmentationTimestamp) {
  tensorflow::SequenceExample sequence;
  std::vector<int> timestamps = {5, 3};
  for (int i = 0; i < timestamps.size(); ++i) {
    AddInstanceSegmentationTimestamp(timestamps[i], &sequence);
    ASSERT_EQ(GetInstanceSegmentationTimestampSize(sequence), i + 1);
    ASSERT_EQ(GetInstanceSegmentationTimestampAt(sequence, i), timestamps[i]);
  }
}

TEST(MediaSequenceTest, RoundTripImageTimestamp) {
  tensorflow::SequenceExample sequence;
  std::vector<int> timestamps = {5, 3};
  for (int i = 0; i < timestamps.size(); ++i) {
    AddImageTimestamp(timestamps[i], &sequence);
    ASSERT_EQ(GetImageTimestampSize(sequence), i + 1);
    ASSERT_EQ(GetImageTimestampAt(sequence, i), timestamps[i]);
  }
  ClearImageTimestamp(&sequence);
  ASSERT_EQ(GetImageTimestampSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripImageFrameRate) {
  tensorflow::SequenceExample sequence;
  double frame_rate = 1.0;
  SetImageFrameRate(frame_rate, &sequence);
  ASSERT_EQ(GetImageFrameRate(sequence), frame_rate);
}

TEST(MediaSequenceTest, RoundTripImageDataPath) {
  tensorflow::SequenceExample sequence;
  std::string data_path = "test";
  SetImageDataPath(data_path, &sequence);
  ASSERT_EQ(data_path, GetImageDataPath(sequence));
}

TEST(MediaSequenceTest, RoundTripFeatureFloats) {
  tensorflow::SequenceExample sequence;
  int num_features = 3;
  int num_floats_in_feature = 4;
  std::string feature_key = "TEST";
  for (int i = 0; i < num_features; ++i) {
    std::vector<float> vf(num_floats_in_feature, 2 << i);
    AddFeatureFloats(feature_key, vf, &sequence);
    ASSERT_EQ(GetFeatureFloatsSize(feature_key, sequence), i + 1);
    for (float value : GetFeatureFloatsAt(feature_key, sequence, i)) {
      ASSERT_EQ(value, 2 << i);
    }
  }
  ClearFeatureFloats(feature_key, &sequence);
  ASSERT_EQ(GetFeatureFloatsSize(feature_key, sequence), 0);
}

TEST(MediaSequenceTest, RoundTripFeatureTimestamp) {
  tensorflow::SequenceExample sequence;
  std::vector<int> timestamps = {5, 3};
  std::string feature_key = "TEST";
  for (int i = 0; i < timestamps.size(); ++i) {
    AddFeatureTimestamp(feature_key, timestamps[i], &sequence);
    ASSERT_EQ(GetFeatureTimestampSize(feature_key, sequence), i + 1);
    ASSERT_EQ(GetFeatureTimestampAt(feature_key, sequence, i), timestamps[i]);
  }
  ClearFeatureTimestamp(feature_key, &sequence);
  ASSERT_EQ(GetFeatureTimestampSize(feature_key, sequence), 0);
}

TEST(MediaSequenceTest, RoundTripContextFeatureFloats) {
  tensorflow::SequenceExample sequence;
  std::string feature_key = "TEST";
  std::vector<float> vf = {0., 1., 2., 4.};
  SetContextFeatureFloats(feature_key, vf, &sequence);
  ASSERT_EQ(GetContextFeatureFloats(feature_key, sequence).size(), vf.size());
  ASSERT_EQ(GetContextFeatureFloats(feature_key, sequence)[3], vf[3]);
  ClearContextFeatureFloats(feature_key, &sequence);
  ASSERT_FALSE(HasFeatureFloats(feature_key, sequence));
}

TEST(MediaSequenceTest, RoundTripContextFeatureBytes) {
  tensorflow::SequenceExample sequence;
  std::string feature_key = "TEST";
  std::vector<std::string> vs = {"0", "1", "2", "4"};
  SetContextFeatureBytes(feature_key, vs, &sequence);
  ASSERT_EQ(GetContextFeatureBytes(feature_key, sequence).size(), vs.size());
  ASSERT_EQ(GetContextFeatureBytes(feature_key, sequence)[3], vs[3]);
  ClearContextFeatureBytes(feature_key, &sequence);
  ASSERT_FALSE(HasFeatureBytes(feature_key, sequence));
}

TEST(MediaSequenceTest, RoundTripContextFeatureInts) {
  tensorflow::SequenceExample sequence;
  std::string feature_key = "TEST";
  std::vector<int64_t> vi = {0, 1, 2, 4};
  SetContextFeatureInts(feature_key, vi, &sequence);
  ASSERT_EQ(GetContextFeatureInts(feature_key, sequence).size(), vi.size());
  ASSERT_EQ(GetContextFeatureInts(feature_key, sequence)[3], vi[3]);
  ClearContextFeatureInts(feature_key, &sequence);
  ASSERT_FALSE(HasFeatureInts(feature_key, sequence));
}

TEST(MediaSequenceTest, RoundTripOpticalFlowEncoded) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> flow = {"test", "again"};
  for (int i = 0; i < flow.size(); ++i) {
    AddForwardFlowEncoded(flow[i], &sequence);
    ASSERT_EQ(GetForwardFlowEncodedSize(sequence), i + 1);
    ASSERT_EQ(GetForwardFlowEncodedAt(sequence, i), flow[i]);
  }
  ClearForwardFlowEncoded(&sequence);
  ASSERT_EQ(GetForwardFlowEncodedSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripOpticalFlowTimestamp) {
  tensorflow::SequenceExample sequence;
  std::vector<int> timestamps = {5, 3};
  for (int i = 0; i < timestamps.size(); ++i) {
    AddForwardFlowTimestamp(timestamps[i], &sequence);
    ASSERT_EQ(GetForwardFlowTimestampSize(sequence), i + 1);
    ASSERT_EQ(GetForwardFlowTimestampAt(sequence, i), timestamps[i]);
  }
  ClearForwardFlowTimestamp(&sequence);
  ASSERT_EQ(GetForwardFlowTimestampSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripTextLanguage) {
  tensorflow::SequenceExample sequence;
  ASSERT_FALSE(HasTextLanguage(sequence));
  SetTextLanguage("test", &sequence);
  ASSERT_TRUE(HasTextLanguage(sequence));
  ASSERT_EQ("test", GetTextLanguage(sequence));
  ClearTextLanguage(&sequence);
  ASSERT_FALSE(HasTextLanguage(sequence));
}

TEST(MediaSequenceTest, RoundTripTextContextContent) {
  tensorflow::SequenceExample sequence;
  ASSERT_FALSE(HasTextContextContent(sequence));
  SetTextContextContent("test", &sequence);
  ASSERT_TRUE(HasTextContextContent(sequence));
  ASSERT_EQ("test", GetTextContextContent(sequence));
  ClearTextContextContent(&sequence);
  ASSERT_FALSE(HasTextContextContent(sequence));
}

TEST(MediaSequenceTest, RoundTripTextContextTokenId) {
  tensorflow::SequenceExample sequence;
  ASSERT_FALSE(HasTextContextTokenId(sequence));
  std::vector<int64_t> vi = {47, 35};
  SetTextContextTokenId(vi, &sequence);
  ASSERT_TRUE(HasTextContextTokenId(sequence));
  ASSERT_EQ(GetTextContextTokenId(sequence).size(), vi.size());
  ASSERT_EQ(GetTextContextTokenId(sequence)[1], vi[1]);
  ClearTextContextTokenId(&sequence);
  ASSERT_FALSE(HasTextContextTokenId(sequence));
}

TEST(MediaSequenceTest, RoundTripTextContextEmbedding) {
  tensorflow::SequenceExample sequence;
  ASSERT_FALSE(HasTextContextEmbedding(sequence));
  std::vector<float> vi = {47., 35.};
  SetTextContextEmbedding(vi, &sequence);
  ASSERT_TRUE(HasTextContextEmbedding(sequence));
  ASSERT_EQ(GetTextContextEmbedding(sequence).size(), vi.size());
  ASSERT_EQ(GetTextContextEmbedding(sequence)[1], vi[1]);
  ClearTextContextEmbedding(&sequence);
  ASSERT_FALSE(HasTextContextEmbedding(sequence));
}

TEST(MediaSequenceTest, RoundTripTextContent) {
  tensorflow::SequenceExample sequence;
  std::vector<std::string> text = {"test", "again"};
  for (int i = 0; i < text.size(); ++i) {
    AddTextContent(text[i], &sequence);
    ASSERT_EQ(GetTextContentSize(sequence), i + 1);
    ASSERT_EQ(GetTextContentAt(sequence, i), text[i]);
  }
  ClearTextContent(&sequence);
  ASSERT_EQ(GetTextContentSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripTextDuration) {
  tensorflow::SequenceExample sequence;
  std::vector<int64_t> timestamps = {4, 7};
  for (int i = 0; i < timestamps.size(); ++i) {
    AddTextTimestamp(timestamps[i], &sequence);
    ASSERT_EQ(GetTextTimestampSize(sequence), i + 1);
    ASSERT_EQ(GetTextTimestampAt(sequence, i), timestamps[i]);
  }
  ClearTextTimestamp(&sequence);
  ASSERT_EQ(GetTextTimestampSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripTextConfidence) {
  tensorflow::SequenceExample sequence;
  std::vector<float> confidence = {0.25, 1.0};
  for (int i = 0; i < confidence.size(); ++i) {
    AddTextConfidence(confidence[i], &sequence);
    ASSERT_EQ(GetTextConfidenceSize(sequence), i + 1);
    ASSERT_EQ(GetTextConfidenceAt(sequence, i), confidence[i]);
  }
  ClearTextConfidence(&sequence);
  ASSERT_EQ(GetTextConfidenceSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripTextEmbedding) {
  tensorflow::SequenceExample sequence;
  int num_features = 3;
  int num_floats_in_feature = 4;
  for (int i = 0; i < num_features; ++i) {
    std::vector<float> vf(num_floats_in_feature, 2 << i);
    AddTextEmbedding(vf, &sequence);
    ASSERT_EQ(GetTextEmbeddingSize(sequence), i + 1);
    for (float value : GetTextEmbeddingAt(sequence, i)) {
      ASSERT_EQ(value, 2 << i);
    }
  }
  ClearTextEmbedding(&sequence);
  ASSERT_EQ(GetTextEmbeddingSize(sequence), 0);
}

TEST(MediaSequenceTest, RoundTripTextTokenId) {
  tensorflow::SequenceExample sequence;
  std::vector<int64_t> ids = {4, 7};
  for (int i = 0; i < ids.size(); ++i) {
    AddTextTokenId(ids[i], &sequence);
    ASSERT_EQ(GetTextTokenIdSize(sequence), i + 1);
    ASSERT_EQ(GetTextTokenIdAt(sequence, i), ids[i]);
  }
  ClearTextTokenId(&sequence);
  ASSERT_EQ(GetTextTokenIdSize(sequence), 0);
}

TEST(MediaSequenceTest, ReconcileMetadataOnEmptySequence) {
  tensorflow::SequenceExample sequence;
  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
}

TEST(MediaSequenceTest, ReconcileMetadataImagestoLabels) {
  // Need image timestamps and label timestamps.
  tensorflow::SequenceExample sequence;
  SetSegmentStartTimestamp(::std::vector<int64_t>({3, 4}), &sequence);
  SetSegmentEndTimestamp(::std::vector<int64_t>({4, 5}), &sequence);

  // Skip 0, so the indices are the timestamp - 1
  AddImageTimestamp(1, &sequence);
  AddImageTimestamp(2, &sequence);
  AddImageTimestamp(3, &sequence);
  AddImageTimestamp(4, &sequence);
  AddImageTimestamp(5, &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_THAT(GetSegmentStartIndex(sequence),
              testing::ElementsAreArray({2, 3}));
  ASSERT_THAT(GetSegmentEndIndex(sequence), testing::ElementsAreArray({3, 4}));
}

TEST(MediaSequenceTest, ReconcileMetadataImages) {
  tensorflow::SequenceExample sequence;
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {}));
  std::string encoded_image(bytes.begin(), bytes.end());
  AddImageEncoded(encoded_image, &sequence);
  AddImageEncoded(encoded_image, &sequence);
  AddImageTimestamp(1000000, &sequence);
  AddImageTimestamp(2000000, &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetContext(sequence, kImageFormatKey).bytes_list().value(0),
            "JPEG");
  ASSERT_EQ(GetContext(sequence, kImageChannelsKey).int64_list().value(0), 3);
  ASSERT_EQ(GetContext(sequence, kImageWidthKey).int64_list().value(0), 3);
  ASSERT_EQ(GetContext(sequence, kImageHeightKey).int64_list().value(0), 2);
  ASSERT_EQ(GetContext(sequence, kImageFrameRateKey).float_list().value(0),
            1.0);
}

TEST(MediaSequenceTest, ReconcileMetadataImagesPNG) {
  tensorflow::SequenceExample sequence;
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(cv::imencode(".png", image, bytes, {}));
  std::string encoded_image(bytes.begin(), bytes.end());
  AddImageEncoded(encoded_image, &sequence);
  AddImageEncoded(encoded_image, &sequence);
  AddImageTimestamp(1000000, &sequence);
  AddImageTimestamp(2000000, &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetContext(sequence, kImageFormatKey).bytes_list().value(0), "PNG");
  ASSERT_EQ(GetContext(sequence, kImageChannelsKey).int64_list().value(0), 3);
  ASSERT_EQ(GetContext(sequence, kImageWidthKey).int64_list().value(0), 3);
  ASSERT_EQ(GetContext(sequence, kImageHeightKey).int64_list().value(0), 2);
  ASSERT_EQ(GetContext(sequence, kImageFrameRateKey).float_list().value(0),
            1.0);
}

TEST(MediaSequenceTest, ReconcileMetadataFlowEncoded) {
  tensorflow::SequenceExample sequence;
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {}));
  std::string encoded_flow(bytes.begin(), bytes.end());

  AddForwardFlowEncoded(encoded_flow, &sequence);
  AddForwardFlowEncoded(encoded_flow, &sequence);
  AddForwardFlowTimestamp(1000000, &sequence);
  AddForwardFlowTimestamp(2000000, &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetForwardFlowFormat(sequence), "JPEG");
  ASSERT_EQ(GetForwardFlowChannels(sequence), 3);
  ASSERT_EQ(GetForwardFlowWidth(sequence), 3);
  ASSERT_EQ(GetForwardFlowHeight(sequence), 2);
  ASSERT_EQ(GetForwardFlowFrameRate(sequence), 1.0);
}

TEST(MediaSequenceTest, ReconcileMetadataFloats) {
  tensorflow::SequenceExample sequence;
  std::vector<float> vf = {3.0, 2.0, 1.0};
  std::string feature_name = "TEST";
  AddFeatureFloats(feature_name, vf, &sequence);
  AddFeatureFloats(feature_name, vf, &sequence);
  AddFeatureTimestamp(feature_name, 1000000, &sequence);
  AddFeatureTimestamp(feature_name, 2000000, &sequence);
  sequence.mutable_feature_lists()->mutable_feature_list()->insert(
      {"EMPTY/feature/floats", tensorflow::FeatureList()});

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetFeatureDimensions(feature_name, sequence).size(), 1);
  ASSERT_EQ(GetFeatureDimensions(feature_name, sequence)[0], 3);
  ASSERT_EQ(GetFeatureRate(feature_name, sequence), 1.0);
}

TEST(MediaSequenceTest, ReconcileMetadataFloatsDoesntOverwrite) {
  tensorflow::SequenceExample sequence;
  std::vector<float> vf = {3.0, 2.0, 1.0, 0.0, -1.0, -2.0};
  std::string feature_name = "TEST";
  SetFeatureDimensions(feature_name, {1, 3, 2}, &sequence);
  AddFeatureFloats(feature_name, vf, &sequence);
  AddFeatureFloats(feature_name, vf, &sequence);
  AddFeatureTimestamp(feature_name, 1000000, &sequence);
  AddFeatureTimestamp(feature_name, 2000000, &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetFeatureDimensions(feature_name, sequence).size(), 3);
  ASSERT_EQ(GetFeatureDimensions(feature_name, sequence)[0], 1);
  ASSERT_EQ(GetFeatureDimensions(feature_name, sequence)[1], 3);
  ASSERT_EQ(GetFeatureDimensions(feature_name, sequence)[2], 2);
  ASSERT_EQ(GetFeatureRate(feature_name, sequence), 1.0);
}

TEST(MediaSequenceTest, ReconcileMetadataFloatsFindsMismatch) {
  tensorflow::SequenceExample sequence;
  std::vector<float> vf = {3.0, 2.0, 1.0, 0.0, -1.0, -2.0};
  std::string feature_name = "TEST";
  SetFeatureDimensions(feature_name, {1, 3, 100}, &sequence);
  AddFeatureFloats(feature_name, vf, &sequence);
  AddFeatureFloats(feature_name, vf, &sequence);
  AddFeatureTimestamp(feature_name, 1000000, &sequence);
  AddFeatureTimestamp(feature_name, 2000000, &sequence);

  ASSERT_FALSE(ReconcileMetadata(true, false, &sequence).ok());
}

TEST(MediaSequenceTest,
     ReconcileMetadataBoxAnnotationsStoresUnmodifiedTimestamps) {
  // Need image timestamps and label timestamps.
  tensorflow::SequenceExample sequence;

  // Skip 0, so the indices are the (timestamp - 10) / 10
  AddImageTimestamp(10, &sequence);
  AddImageTimestamp(20, &sequence);
  AddImageTimestamp(30, &sequence);
  AddImageTimestamp(40, &sequence);

  AddBBoxTimestamp(11, &sequence);
  AddBBoxTimestamp(12, &sequence);  // Will be dropped in the output.
  AddBBoxTimestamp(39, &sequence);

  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.1, 0.2)},
      {Location::CreateRelativeBBoxLocation(0.1, 0.3, 0.5, 0.7)}};
  AddBBox(bboxes[0], &sequence);
  AddBBox(bboxes[1], &sequence);
  AddBBox(bboxes[2], &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));

  ASSERT_EQ(GetBBoxTimestampSize(sequence), 4);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 0), 10);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 1), 20);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 2), 30);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 3), 40);

  ASSERT_EQ(GetBBoxIsAnnotatedSize(sequence), 4);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 0), true);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 1), false);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 2), false);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 3), true);

  // Unmodified timestamp is only stored for is_annotated == true.
  ASSERT_EQ(GetUnmodifiedBBoxTimestampSize(sequence), 2);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt(sequence, 0), 11);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt(sequence, 1), 39);

  // A second reconciliation should not corrupt unmodified bbox timestamps.
  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));

  ASSERT_EQ(GetBBoxTimestampSize(sequence), 4);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 0), 10);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 1), 20);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 2), 30);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 3), 40);

  ASSERT_EQ(GetUnmodifiedBBoxTimestampSize(sequence), 2);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt(sequence, 0), 11);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt(sequence, 1), 39);
}

TEST(MediaSequenceTest, ReconcileMetadataBoxAnnotationsFillsMissing) {
  // Need image timestamps and label timestamps.
  tensorflow::SequenceExample sequence;

  // Skip 0, so the indices are the (timestamp - 10) / 10
  AddImageTimestamp(10, &sequence);
  AddImageTimestamp(20, &sequence);
  AddImageTimestamp(30, &sequence);
  AddImageTimestamp(40, &sequence);
  AddImageTimestamp(50, &sequence);

  AddBBoxTimestamp(9, &sequence);
  AddBBoxTimestamp(21, &sequence);
  AddBBoxTimestamp(22, &sequence);  // Will be dropped in the output.

  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.1, 0.2)},
      {Location::CreateRelativeBBoxLocation(0.1, 0.3, 0.5, 0.7)}};
  AddBBox(bboxes[0], &sequence);
  AddBBox(bboxes[1], &sequence);
  AddBBox(bboxes[2], &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetBBoxTimestampSize(sequence), 5);
  ASSERT_EQ(GetBBoxIsAnnotatedSize(sequence), 5);

  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 0), true);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 1), true);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 2), false);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 3), false);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 4), false);

  ASSERT_EQ(GetBBoxTimestampAt(sequence, 0), 10);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 1), 20);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 2), 30);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 3), 40);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 4), 50);

  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 0), 1);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 1), 1);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 2), 0);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 3), 0);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 4), 0);
}

TEST(MediaSequenceTest, ReconcileMetadataBoxAnnotationsUpdatesAllFeatures) {
  // Need image timestamps and label timestamps.
  tensorflow::SequenceExample sequence;

  // Skip 0, so the indices are the (timestamp - 10) / 10
  AddImageTimestamp(10, &sequence);
  AddImageTimestamp(20, &sequence);
  AddImageTimestamp(30, &sequence);
  AddImageTimestamp(40, &sequence);
  AddImageTimestamp(50, &sequence);

  AddBBoxTimestamp(9, &sequence);
  AddBBoxTimestamp(21, &sequence);

  AddBBoxNumRegions(1, &sequence);
  AddBBoxNumRegions(1, &sequence);

  AddBBoxLabelIndex(::std::vector<int64_t>({1}), &sequence);
  AddBBoxLabelIndex(::std::vector<int64_t>({2}), &sequence);

  AddBBoxLabelString(::std::vector<std::string>({"one"}), &sequence);
  AddBBoxLabelString(::std::vector<std::string>({"two"}), &sequence);

  AddBBoxClassIndex(::std::vector<int64_t>({1}), &sequence);
  AddBBoxClassIndex(::std::vector<int64_t>({2}), &sequence);

  AddBBoxClassString(::std::vector<std::string>({"one"}), &sequence);
  AddBBoxClassString(::std::vector<std::string>({"two"}), &sequence);

  AddBBoxTrackIndex(::std::vector<int64_t>({1}), &sequence);
  AddBBoxTrackIndex(::std::vector<int64_t>({2}), &sequence);

  AddBBoxTrackString(::std::vector<std::string>({"one"}), &sequence);
  AddBBoxTrackString(::std::vector<std::string>({"two"}), &sequence);

  ::std::vector<::std::vector<::std::pair<float, float>>> points = {
      {{0.35, 0.47}}, {{0.47, 0.35}}};
  AddBBoxPoint(points[0], &sequence);
  AddBBoxPoint(points[1], &sequence);

  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.1, 0.2)}};
  AddBBox(bboxes[0], &sequence);
  AddBBox(bboxes[1], &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetBBoxTimestampSize(sequence), 5);
  ASSERT_EQ(GetBBoxIsAnnotatedSize(sequence), 5);

  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 0), true);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 1), true);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 2), false);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 3), false);
  ASSERT_EQ(GetBBoxIsAnnotatedAt(sequence, 4), false);

  ASSERT_EQ(GetBBoxTimestampAt(sequence, 0), 10);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 1), 20);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 2), 30);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 3), 40);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 4), 50);

  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 0), 1);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 1), 1);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 2), 0);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 3), 0);
  ASSERT_EQ(GetBBoxNumRegionsAt(sequence, 4), 0);

  ASSERT_THAT(GetBBoxLabelIndexAt(sequence, 0),
              ::testing::ElementsAreArray({1}));
  ASSERT_THAT(GetBBoxLabelIndexAt(sequence, 1),
              ::testing::ElementsAreArray({2}));
  ASSERT_THAT(GetBBoxLabelIndexAt(sequence, 2),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));
  ASSERT_THAT(GetBBoxLabelIndexAt(sequence, 3),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));
  ASSERT_THAT(GetBBoxLabelIndexAt(sequence, 4),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));

  ASSERT_THAT(GetBBoxLabelStringAt(sequence, 0),
              ::testing::ElementsAreArray({"one"}));
  ASSERT_THAT(GetBBoxLabelStringAt(sequence, 1),
              ::testing::ElementsAreArray({"two"}));
  ASSERT_THAT(GetBBoxLabelStringAt(sequence, 2),
              ::testing::ElementsAreArray(::std::vector<std::string>()));
  ASSERT_THAT(GetBBoxLabelStringAt(sequence, 3),
              ::testing::ElementsAreArray(::std::vector<std::string>()));
  ASSERT_THAT(GetBBoxLabelStringAt(sequence, 4),
              ::testing::ElementsAreArray(::std::vector<std::string>()));

  ASSERT_THAT(GetBBoxClassIndexAt(sequence, 0),
              ::testing::ElementsAreArray({1}));
  ASSERT_THAT(GetBBoxClassIndexAt(sequence, 1),
              ::testing::ElementsAreArray({2}));
  ASSERT_THAT(GetBBoxClassIndexAt(sequence, 2),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));
  ASSERT_THAT(GetBBoxClassIndexAt(sequence, 3),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));
  ASSERT_THAT(GetBBoxClassIndexAt(sequence, 4),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));

  ASSERT_THAT(GetBBoxClassStringAt(sequence, 0),
              ::testing::ElementsAreArray({"one"}));
  ASSERT_THAT(GetBBoxClassStringAt(sequence, 1),
              ::testing::ElementsAreArray({"two"}));
  ASSERT_THAT(GetBBoxClassStringAt(sequence, 2),
              ::testing::ElementsAreArray(::std::vector<std::string>()));
  ASSERT_THAT(GetBBoxClassStringAt(sequence, 3),
              ::testing::ElementsAreArray(::std::vector<std::string>()));
  ASSERT_THAT(GetBBoxClassStringAt(sequence, 4),
              ::testing::ElementsAreArray(::std::vector<std::string>()));

  ASSERT_THAT(GetBBoxTrackIndexAt(sequence, 0),
              ::testing::ElementsAreArray({1}));
  ASSERT_THAT(GetBBoxTrackIndexAt(sequence, 1),
              ::testing::ElementsAreArray({2}));
  ASSERT_THAT(GetBBoxTrackIndexAt(sequence, 2),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));
  ASSERT_THAT(GetBBoxTrackIndexAt(sequence, 3),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));
  ASSERT_THAT(GetBBoxTrackIndexAt(sequence, 4),
              ::testing::ElementsAreArray(::std::vector<int64_t>()));

  ASSERT_THAT(GetBBoxTrackStringAt(sequence, 0),
              ::testing::ElementsAreArray({"one"}));
  ASSERT_THAT(GetBBoxTrackStringAt(sequence, 1),
              ::testing::ElementsAreArray({"two"}));
  ASSERT_THAT(GetBBoxTrackStringAt(sequence, 2),
              ::testing::ElementsAreArray(::std::vector<std::string>()));
  ASSERT_THAT(GetBBoxTrackStringAt(sequence, 3),
              ::testing::ElementsAreArray(::std::vector<std::string>()));
  ASSERT_THAT(GetBBoxTrackStringAt(sequence, 4),
              ::testing::ElementsAreArray(::std::vector<std::string>()));

  EXPECT_EQ(bboxes[0].size(), GetBBoxAt(sequence, 0).size());
  EXPECT_EQ(GetBBoxAt(sequence, 0)[0].GetRelativeBBox(),
            bboxes[0][0].GetRelativeBBox());
  EXPECT_EQ(bboxes[1].size(), GetBBoxAt(sequence, 1).size());
  EXPECT_EQ(GetBBoxAt(sequence, 1)[0].GetRelativeBBox(),
            bboxes[1][0].GetRelativeBBox());
  EXPECT_EQ(0, GetBBoxAt(sequence, 2).size());
  EXPECT_EQ(0, GetBBoxAt(sequence, 3).size());
  EXPECT_EQ(0, GetBBoxAt(sequence, 4).size());

  EXPECT_EQ(1, GetBBoxPointAt(sequence, 0).size());
  EXPECT_EQ(points[0][0], GetBBoxPointAt(sequence, 0)[0]);
  EXPECT_EQ(1, GetBBoxPointAt(sequence, 1).size());
  EXPECT_EQ(points[1][0], GetBBoxPointAt(sequence, 1)[0]);
  EXPECT_EQ(0, GetBBoxPointAt(sequence, 2).size());
  EXPECT_EQ(0, GetBBoxPointAt(sequence, 3).size());
  EXPECT_EQ(0, GetBBoxPointAt(sequence, 4).size());
}

TEST(MediaSequenceTest, ReconcileMetadataBoxAnnotationsDoesNotAddFields) {
  // Need image timestamps and label timestamps.
  tensorflow::SequenceExample sequence;

  // Skip 0, so the indices are the (timestamp - 10) / 10
  AddImageTimestamp(10, &sequence);
  AddImageTimestamp(20, &sequence);
  AddImageTimestamp(30, &sequence);
  AddImageTimestamp(40, &sequence);
  AddImageTimestamp(50, &sequence);

  AddBBoxTimestamp(9, &sequence);
  AddBBoxTimestamp(21, &sequence);
  AddBBoxTimestamp(22, &sequence);  // Will be dropped in the output.

  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.1, 0.2)},
      {Location::CreateRelativeBBoxLocation(0.1, 0.3, 0.5, 0.7)}};
  AddBBox(bboxes[0], &sequence);
  AddBBox(bboxes[1], &sequence);
  AddBBox(bboxes[2], &sequence);

  MP_ASSERT_OK(ReconcileMetadata(true, false, &sequence));
  ASSERT_EQ(GetBBoxTimestampSize(sequence), 5);
  ASSERT_EQ(GetBBoxIsAnnotatedSize(sequence), 5);
  ASSERT_FALSE(HasBBoxClassIndex(sequence));
  ASSERT_FALSE(HasBBoxLabelIndex(sequence));
  ASSERT_FALSE(HasBBoxLabelString(sequence));
  ASSERT_FALSE(HasBBoxClassString(sequence));
  ASSERT_FALSE(HasBBoxTrackString(sequence));
}

TEST(MediaSequenceTest, ReconcileMetadataRegionAnnotations) {
  // Need image timestamps and label timestamps.
  tensorflow::SequenceExample sequence;

  // Skip 0, so the indices are the (timestamp - 10) / 10
  AddImageTimestamp(10, &sequence);
  AddImageTimestamp(20, &sequence);
  AddImageTimestamp(30, &sequence);

  AddBBoxTimestamp(9, &sequence);
  AddBBoxTimestamp(21, &sequence);
  AddBBoxTimestamp(22, &sequence);  // Will be dropped in the output.

  AddBBoxTimestamp("PREFIX", 8, &sequence);  // Will be dropped in the output.
  AddBBoxTimestamp("PREFIX", 9, &sequence);
  AddBBoxTimestamp("PREFIX", 22, &sequence);

  // Expect both the default and "PREFIX"-ed keys to be reconciled.
  MP_ASSERT_OK(ReconcileMetadata(false, true, &sequence));
  ASSERT_EQ(GetBBoxTimestampSize(sequence), 3);
  ASSERT_EQ(GetBBoxIsAnnotatedSize(sequence), 3);
  ASSERT_EQ(GetBBoxTimestampSize("PREFIX", sequence), 3);
  ASSERT_EQ(GetBBoxIsAnnotatedSize("PREFIX", sequence), 3);

  ASSERT_EQ(GetBBoxTimestampAt(sequence, 0), 10);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 1), 20);
  ASSERT_EQ(GetBBoxTimestampAt(sequence, 2), 30);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampSize(sequence), 2);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt(sequence, 0), 9);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt(sequence, 1), 21);

  ASSERT_EQ(GetBBoxTimestampAt("PREFIX", sequence, 0), 10);
  ASSERT_EQ(GetBBoxTimestampAt("PREFIX", sequence, 1), 20);
  ASSERT_EQ(GetBBoxTimestampAt("PREFIX", sequence, 2), 30);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampSize("PREFIX", sequence), 2);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt("PREFIX", sequence, 0), 9);
  ASSERT_EQ(GetUnmodifiedBBoxTimestampAt("PREFIX", sequence, 1), 22);
}
}  // namespace
}  // namespace mediasequence
}  // namespace mediapipe
