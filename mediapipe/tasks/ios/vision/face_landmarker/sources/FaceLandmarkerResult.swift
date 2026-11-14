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
package import third_party_mediapipe_tasks_c_components_containers_category
package import third_party_mediapipe_tasks_c_components_containers_landmark
package import third_party_mediapipe_tasks_c_components_containers_matrix
package import third_party_mediapipe_tasks_c_vision_face_landmarker_face_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// A dense matrix of `Float` values stored in column-major order.
public struct TransformMatrix: Sendable, Equatable, Hashable {
  /// Number of rows.
  public let rows: Int
  /// Number of columns.
  public let columns: Int
  /// Matrix values in column-major order, containing `rows * columns` elements.
  public let data: [Float]

  /// Creates a matrix from `rows * columns` column-major `data` values.
  ///
  /// - Throws: `TasksError` if `rows` or `columns` is negative, `rows * columns` overflows, or
  ///   `data.count` does not equal `rows * columns`.
  public init(data: [Float], rows: Int, columns: Int) throws {
    let (expectedCount, overflow) = rows.multipliedReportingOverflow(by: columns)
    guard rows >= 0, columns >= 0, !overflow, data.count == expectedCount else {
      throw TasksError.invalidArgument(
        "TransformMatrix data count (\(data.count)) must equal rows * columns (\(rows) * \(columns))."
      )
    }
    self.rows = rows
    self.columns = columns
    self.data = data
  }

  /// Creates a matrix by copying a C `MpMatrix` struct.
  package init(cMatrix: MpMatrix) {
    let (rows, columns) = (Int(cMatrix.rows), Int(cMatrix.cols))
    precondition(rows >= 0 && columns >= 0)
    self.rows = rows
    self.columns = columns
    self.data = Array(cArray: cMatrix.data, count: rows * columns) { $0 }
  }

  /// Returns the value at `row` and `column`. Both must be within the matrix bounds.
  public func value(atRow row: Int, column: Int) -> Float {
    precondition((0..<rows).contains(row) && (0..<columns).contains(column))
    return data[column * rows + row]
  }
}

/// Face landmarks, blendshapes, and transformation matrixes produced by `FaceLandmarker`, with
/// one element per detected face.
public struct FaceLandmarkerResult: Sendable, Equatable, Hashable {
  /// 478 normalized face mesh landmarks per detected face.
  public let faceLandmarks: [[NormalizedLandmark]]
  /// Blendshape scores per detected face (`*Left`/`*Right` refer to the subject, not the viewer),
  /// or empty if blendshapes were not requested.
  public let faceBlendshapes: [Classifications]
  /// 4x4 facial transformation matrixes per detected face, or empty if not requested.
  public let facialTransformationMatrixes: [TransformMatrix]
  /// Timestamp in milliseconds of the input frame, or `0` in `.image` mode.
  public let timestampInMilliseconds: Int

  /// Creates a face landmarker result with per-face outputs and timestamp.
  public init(
    faceLandmarks: [[NormalizedLandmark]],
    faceBlendshapes: [Classifications],
    facialTransformationMatrixes: [TransformMatrix],
    timestampInMilliseconds: Int
  ) {
    self.faceLandmarks = faceLandmarks
    self.faceBlendshapes = faceBlendshapes
    self.facialTransformationMatrixes = facialTransformationMatrixes
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates a result by deep-copying a C `MpFaceLandmarkerResult`.
  package init(cResult: MpFaceLandmarkerResult, timestampInMilliseconds: Int) {
    self.faceLandmarks = Array(
      cArray: cResult.face_landmarks, count: cResult.face_landmarks_count,
      transform: NormalizedLandmark.landmarks(from:))
    self.faceBlendshapes = Array(
      cArray: cResult.face_blendshapes, count: cResult.face_blendshapes_count
    ) { cCategories in
      let categories = Array(
        cArray: cCategories.categories, count: cCategories.categories_count,
        transform: ResultCategory.init)
      return Classifications(headIndex: 0, categories: categories)
    }
    self.facialTransformationMatrixes = Array(
      cArray: cResult.facial_transformation_matrixes,
      count: cResult.facial_transformation_matrixes_count, transform: TransformMatrix.init)
    self.timestampInMilliseconds = timestampInMilliseconds
  }
}
