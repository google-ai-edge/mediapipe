/**
 * Copyright 2022 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {FilesetResolver as FilesetResolverImpl} from '../../../tasks/web/core/fileset_resolver';
import {DrawingUtils as DrawingUtilsImpl} from '../../../tasks/web/vision/core/drawing_utils';
import {MPImage as MPImageImpl} from '../../../tasks/web/vision/core/image';
import {MPMask as MPMaskImpl} from '../../../tasks/web/vision/core/mask';
import {FaceDetector as FaceDetectorImpl} from '../../../tasks/web/vision/face_detector/face_detector';
import {FaceLandmarker as FaceLandmarkerImpl} from '../../../tasks/web/vision/face_landmarker/face_landmarker';
import {FaceStylizer as FaceStylizerImpl} from '../../../tasks/web/vision/face_stylizer/face_stylizer';
import {GestureRecognizer as GestureRecognizerImpl} from '../../../tasks/web/vision/gesture_recognizer/gesture_recognizer';
import {HandLandmarker as HandLandmarkerImpl} from '../../../tasks/web/vision/hand_landmarker/hand_landmarker';
import {HolisticLandmarker as HolisticLandmarkerImpl} from '../../../tasks/web/vision/holistic_landmarker/holistic_landmarker';
import {ImageClassifier as ImageClassifierImpl} from '../../../tasks/web/vision/image_classifier/image_classifier';
import {ImageEmbedder as ImageEmbedderImpl} from '../../../tasks/web/vision/image_embedder/image_embedder';
import {ImageSegmenter as ImageSegementerImpl} from '../../../tasks/web/vision/image_segmenter/image_segmenter';
import {InteractiveSegmenter as InteractiveSegmenterImpl} from '../../../tasks/web/vision/interactive_segmenter/interactive_segmenter';
import {ObjectDetector as ObjectDetectorImpl} from '../../../tasks/web/vision/object_detector/object_detector';
import {PoseLandmarker as PoseLandmarkerImpl} from '../../../tasks/web/vision/pose_landmarker/pose_landmarker';

// Declare the variables locally so that Rollup in OSS includes them explicitly
// as exports.
const DrawingUtils = DrawingUtilsImpl;
const FilesetResolver = FilesetResolverImpl;
const MPImage = MPImageImpl;
const MPMask = MPMaskImpl;
const FaceDetector = FaceDetectorImpl;
const FaceLandmarker = FaceLandmarkerImpl;
const FaceStylizer = FaceStylizerImpl;
const GestureRecognizer = GestureRecognizerImpl;
const HandLandmarker = HandLandmarkerImpl;
const HolisticLandmarker = HolisticLandmarkerImpl;
const ImageClassifier = ImageClassifierImpl;
const ImageEmbedder = ImageEmbedderImpl;
const ImageSegmenter = ImageSegementerImpl;
const InteractiveSegmenter = InteractiveSegmenterImpl;
const ObjectDetector = ObjectDetectorImpl;
const PoseLandmarker = PoseLandmarkerImpl;

export {
  DrawingUtils,
  FilesetResolver,
  MPImage,
  MPMask,
  FaceDetector,
  FaceLandmarker,
  FaceStylizer,
  GestureRecognizer,
  HandLandmarker,
  HolisticLandmarker,
  ImageClassifier,
  ImageEmbedder,
  ImageSegmenter,
  InteractiveSegmenter,
  ObjectDetector,
  PoseLandmarker
};
