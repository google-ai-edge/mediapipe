// Copyright 2023 The MediaPipe Authors.
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

#import <XCTest/XCTest.h>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerResult.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerResult+Helpers.h"

using ::mediapipe::MakePacket;
using ::mediapipe::Packet;
using ::mediapipe::Timestamp;
using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
using ClassificationListProto = ::mediapipe::ClassificationList;
using FaceGeometryProto = ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;

static constexpr int kMicrosecondsPerMillisecond = 1000;

@interface MPPLandmarkerResultHelpersTests : XCTestCase {
}
@end

@implementation MPPLandmarkerResultHelpersTests

- (void)testCreatesResultFromLandmarkerPackets {
  const std::vector<NormalizedLandmarkListProto> normalizedLandmarkProtos({{}});
  const std::vector<ClassificationListProto> classificationProtos({{}});
  const std::vector<FaceGeometryProto> faceGeometryProto({{}});

  const auto landmarksPacket =
      MakePacket<std::vector<NormalizedLandmarkListProto>>(normalizedLandmarkProtos)
          .At(Timestamp(42 * kMicrosecondsPerMillisecond));
  const auto classificationsPacket =
      MakePacket<std::vector<ClassificationListProto>>(classificationProtos)
          .At(Timestamp(42 * kMicrosecondsPerMillisecond));
  const auto faceGeometryPacket = MakePacket<std::vector<FaceGeometryProto>>(faceGeometryProto)
                                      .At(Timestamp(42 * kMicrosecondsPerMillisecond));

  MPPFaceLandmarkerResult *results =
      [MPPFaceLandmarkerResult faceLandmarkerResultWithLandmarksPacket:landmarksPacket
                                                     blendshapesPacket:classificationsPacket
                                          transformationMatrixesPacket:faceGeometryPacket];

  XCTAssertEqual(results.faceLandmarks.count, 1);
  XCTAssertEqual(results.faceBlendshapes.count, 1);
  XCTAssertEqual(results.facialTransformationMatrixes.count, 1);
  XCTAssertEqual(results.timestampInMilliseconds, 42);
}

- (void)testCreatesCreatesCopyOfFacialTransformationMatrix {
  MPPFaceLandmarkerResult *results;

  {
    // Create scope so that FaceGeometryProto gets deallocated before we access the
    // MPPFaceLandmarkerResult.
    FaceGeometryProto faceGeometryProto{};
    auto *matrixData = faceGeometryProto.mutable_pose_transform_matrix();
    matrixData->set_cols(4);
    matrixData->set_rows(4);
    for (size_t i = 0; i < 4 * 4; ++i) {
      matrixData->add_packed_data(0.1f * i);
    }

    const std::vector<FaceGeometryProto> faceGeometryProtos({faceGeometryProto});
    const auto faceGeometryPacket = MakePacket<std::vector<FaceGeometryProto>>(faceGeometryProtos);
    results = [MPPFaceLandmarkerResult faceLandmarkerResultWithLandmarksPacket:{}
        blendshapesPacket:{}
        transformationMatrixesPacket:faceGeometryPacket];
  }

  XCTAssertEqual(results.facialTransformationMatrixes.count, 1);
  XCTAssertEqual(results.facialTransformationMatrixes[0].rows, 4);
  XCTAssertEqual(results.facialTransformationMatrixes[0].columns, 4);
  for (size_t column = 0; column < 4; ++column) {
    for (size_t row = 0; row < 4; ++row) {
      XCTAssertEqualWithAccuracy(
          [results.facialTransformationMatrixes[0] valueAtRow:row column:column],
          0.4f * row + 0.1f * column, /* accuracy= */ 0.0001f, @"at [%zu,%zu]", column, row);
    }
  }
}

- (void)testCreatesResultFromEmptyPackets {
  const Packet emptyPacket = Packet{}.At(Timestamp(0));
  MPPFaceLandmarkerResult *results =
      [MPPFaceLandmarkerResult faceLandmarkerResultWithLandmarksPacket:emptyPacket
                                                     blendshapesPacket:emptyPacket
                                          transformationMatrixesPacket:emptyPacket];

  NSArray *emptyArray = [NSArray array];
  XCTAssertEqualObjects(results.faceLandmarks, emptyArray);
  XCTAssertEqualObjects(results.faceBlendshapes, emptyArray);
  XCTAssertEqualObjects(results.facialTransformationMatrixes, emptyArray);
  XCTAssertEqual(results.timestampInMilliseconds, 0);
}

@end
