// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.solutions.facedetection;

import android.content.Context;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.solutioncore.ErrorListener;
import com.google.mediapipe.solutioncore.ImageSolutionBase;
import com.google.mediapipe.solutioncore.OutputHandler;
import com.google.mediapipe.solutioncore.ResultListener;
import com.google.mediapipe.solutioncore.SolutionInfo;
import com.google.mediapipe.formats.proto.DetectionProto.Detection;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * MediaPipe Face Detection Solution API.
 *
 * <p>MediaPipe Face Detection processes a {@link TextureFrame} or a {@link Bitmap} and returns the
 * {@link FaceDetectionResult} representing each detected face. Please refer to
 * https://solutions.mediapipe.dev/face_detection#android-solution-api for usage examples.
 */
public class FaceDetection extends ImageSolutionBase {
  private static final String TAG = "FaceDetection";

  private static final String SHORT_RANGE_GRAPH_NAME = "face_detection_short_range_image.binarypb";
  private static final String FULL_RANGE_GRAPH_NAME = "face_detection_full_range_image.binarypb";
  private static final String IMAGE_INPUT_STREAM = "image";
  private static final ImmutableList<String> OUTPUT_STREAMS =
      ImmutableList.of("detections", "throttled_image");
  private static final int DETECTIONS_INDEX = 0;
  private static final int INPUT_IMAGE_INDEX = 1;
  private final OutputHandler<FaceDetectionResult> outputHandler;

  /**
   * Initializes MediaPipe Face Detection solution.
   *
   * @param context an Android {@link Context}.
   * @param options the configuration options defined in {@link FaceDetectionOptions}.
   */
  public FaceDetection(Context context, FaceDetectionOptions options) {
    outputHandler = new OutputHandler<>();
    outputHandler.setOutputConverter(
        packets -> {
          FaceDetectionResult.Builder faceMeshResultBuilder = FaceDetectionResult.builder();
          try {
            faceMeshResultBuilder.setMultiFaceDetections(
                getProtoVector(packets.get(DETECTIONS_INDEX), Detection.parser()));
          } catch (MediaPipeException e) {
            reportError("Error occurs while getting MediaPipe face detection results.", e);
          }
          return faceMeshResultBuilder
              .setImagePacket(packets.get(INPUT_IMAGE_INDEX))
              .setTimestamp(
                  staticImageMode ? Long.MIN_VALUE : packets.get(INPUT_IMAGE_INDEX).getTimestamp())
              .build();
        });

    SolutionInfo solutionInfo =
        SolutionInfo.builder()
            .setBinaryGraphPath(
                options.modelSelection() == 0 ? SHORT_RANGE_GRAPH_NAME : FULL_RANGE_GRAPH_NAME)
            .setImageInputStreamName(IMAGE_INPUT_STREAM)
            .setOutputStreamNames(OUTPUT_STREAMS)
            .setStaticImageMode(options.staticImageMode())
            .build();

    initialize(context, solutionInfo, outputHandler);
    Map<String, Packet> emptyInputSidePackets = new HashMap<>();
    start(emptyInputSidePackets);
  }

  /**
   * Sets a callback to be invoked when a {@link FaceDetectionResult} becomes available.
   *
   * @param listener the {@link ResultListener} callback.
   */
  public void setResultListener(ResultListener<FaceDetectionResult> listener) {
    this.outputHandler.setResultListener(listener);
  }

  /**
   * Sets a callback to be invoked when the Face Detection solution throws errors.
   *
   * @param listener the {@link ErrorListener} callback.
   */
  public void setErrorListener(@Nullable ErrorListener listener) {
    this.outputHandler.setErrorListener(listener);
    this.errorListener = listener;
  }
}
