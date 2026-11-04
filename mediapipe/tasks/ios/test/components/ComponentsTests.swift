// Copyright 2026 The MediaPipe Authors.
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

import Foundation
import MediaPipeTasksComponents
import MediaPipeTasksCore
import Testing
import third_party_mediapipe_tasks_c_components_containers_category
import third_party_mediapipe_tasks_c_components_containers_classification_result
import third_party_mediapipe_tasks_c_components_containers_detection_result
import third_party_mediapipe_tasks_c_components_containers_keypoint
import third_party_mediapipe_tasks_c_components_containers_landmark
import third_party_mediapipe_tasks_c_components_containers_rect

#if canImport(CoreGraphics)
  import CoreGraphics
#endif

struct ComponentsTests {
  @Test
  func `ResultCategory and ClassificationResult bridge from C structs`() throws {
    var categoryName = Array("music".utf8CString)
    var displayName = Array("Music".utf8CString)
    var headName = Array("audio_head".utf8CString)

    try categoryName.withUnsafeMutableBufferPointer { categoryNameBuf in
      try displayName.withUnsafeMutableBufferPointer { displayNameBuf in
        try headName.withUnsafeMutableBufferPointer { headNameBuf in
          var cCategory = MpCategory(
            index: 2, score: 0.92, category_name: categoryNameBuf.baseAddress,
            display_name: displayNameBuf.baseAddress)
          var cClassification = MpClassifications(
            categories: &cCategory, categories_count: 1, head_index: 0,
            head_name: headNameBuf.baseAddress)
          let cResult = MpClassificationResult(
            classifications: &cClassification, classifications_count: 1, timestamp_ms: 1500,
            has_timestamp_ms: true)

          let result = ClassificationResult(cResult: cResult)
          #expect(result.timestampInMilliseconds == 1500)
          let classification = try #require(result.classifications.first)
          #expect(classification.headIndex == 0 && classification.headName == "audio_head")
          let category = try #require(classification.categories.first)
          #expect(
            category
              == ResultCategory(index: 2, score: 0.92, categoryName: "music", displayName: "Music"))
        }
      }
    }
  }

  @Test
  func `DetectionResult bridges bounding box and keypoints and hashes consistently`() throws {
    var keypointLabel = Array("left_eye".utf8CString)
    try keypointLabel.withUnsafeMutableBufferPointer { labelBuf in
      var cKeypoint = MpNormalizedKeypoint(
        x: 0.25, y: 0.5, label: labelBuf.baseAddress, score: 0.95, has_score: true
      )
      var cDetection = MpDetection(
        categories: nil, categories_count: 0,
        bounding_box: MpRect(left: 10, top: 20, bottom: 70, right: 50),
        keypoints: &cKeypoint, keypoints_count: 1
      )
      let cResult = MpDetectionResult(detections: &cDetection, detections_count: 1)

      let result = DetectionResult(cResult: cResult)
      let detection = try #require(result.detections.first)
      #expect(detection.boundingBox == CGRect(x: 10, y: 20, width: 40, height: 50))
      let keypoint = try #require(detection.keypoints?.first)
      let expectedKeypoint = NormalizedKeypoint(
        location: CGPoint(x: 0.25, y: 0.5), label: "left_eye", score: 0.95
      )
      #expect(keypoint == expectedKeypoint)
    }

    let standard = Detection(categories: [], boundingBox: CGRect(x: 5, y: 5, width: 5, height: 5))
    let inverted = Detection(
      categories: [], boundingBox: CGRect(x: 10, y: 10, width: -5, height: -5))
    #expect(standard == inverted && standard.hashValue == inverted.hashValue)
  }

  @Test
  func `Landmark, NormalizedLandmark, Connection, and RegionOfInterest preserve value semantics`() {
    var cLandmark = MpLandmark(
      x: 1.0, y: 2.0, z: 3.0,
      has_visibility: true, visibility: 0.9, has_presence: false, presence: 0.0, name: nil
    )
    let landmark = Landmark(cLandmark: cLandmark)
    #expect(landmark == Landmark(x: 1.0, y: 2.0, z: 3.0, visibility: 0.9))
    #expect(
      Landmark.landmarks(from: MpLandmarks(landmarks: &cLandmark, landmarks_count: 1))
        == [landmark]
    )

    var cNormalized = MpNormalizedLandmark(
      x: 0.5, y: 0.5, z: 0.1,
      has_visibility: false, visibility: 0.0, has_presence: true, presence: 0.8, name: nil
    )
    let normalized = NormalizedLandmark(cLandmark: cNormalized)
    #expect(normalized == NormalizedLandmark(x: 0.5, y: 0.5, z: 0.1, presence: 0.8))
    #expect(
      NormalizedLandmark.landmarks(
        from: MpNormalizedLandmarks(landmarks: &cNormalized, landmarks_count: 1)
      ) == [normalized]
    )

    #expect(Connection(start: 0, end: 1) == Connection(start: 0, end: 1))
    let point = NormalizedKeypoint(location: CGPoint(x: 0.3, y: 0.4))
    #expect(RegionOfInterest(keypoint: point).keypoint == point)
    #expect(RegionOfInterest(scribbles: [point]).scribbles == [point])
  }
}
