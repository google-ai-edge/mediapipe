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

package com.google.mediapipe.solutions.facemesh;

import android.content.Context;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.solutioncore.ErrorListener;
import com.google.mediapipe.solutioncore.ImageSolutionBase;
import com.google.mediapipe.solutioncore.OutputHandler;
import com.google.mediapipe.solutioncore.ResultListener;
import com.google.mediapipe.solutioncore.SolutionInfo;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * MediaPipe Face Mesh Solution API.
 *
 * <p>MediaPipe Face Mesh processes a {@link TextureFrame} or a {@link Bitmap} and returns the face
 * landmarks of each detected face. Please refer to
 * https://solutions.mediapipe.dev/face_mesh#android-solution-api for usage examples.
 */
public class FaceMesh extends ImageSolutionBase {
  private static final String TAG = "FaceMesh";

  public static final int FACEMESH_NUM_LANDMARKS = 468;
  public static final int FACEMESH_NUM_LANDMARKS_WITH_IRISES = 478;

  private static final String NUM_FACES = "num_faces";
  private static final String WITH_ATTENTION = "with_attention";
  private static final String USE_PREV_LANDMARKS = "use_prev_landmarks";
  private static final String GPU_GRAPH_NAME = "face_landmark_front_gpu_image.binarypb";
  private static final String CPU_GRAPH_NAME = "face_landmark_front_cpu_image.binarypb";
  private static final String IMAGE_INPUT_STREAM = "image";
  private static final ImmutableList<String> OUTPUT_STREAMS =
      ImmutableList.of("multi_face_landmarks", "throttled_image");
  private static final int LANDMARKS_INDEX = 0;
  private static final int INPUT_IMAGE_INDEX = 1;
  private final OutputHandler<FaceMeshResult> outputHandler;

  /**
   * Initializes MediaPipe Face Mesh solution.
   *
   * @param context an Android {@link Context}.
   * @param options the configuration options defined in {@link FaceMeshOptions}.
   */
  public FaceMesh(Context context, FaceMeshOptions options) {
    outputHandler = new OutputHandler<>();
    outputHandler.setOutputConverter(
        packets -> {
          FaceMeshResult.Builder faceMeshResultBuilder = FaceMeshResult.builder();
          try {
            faceMeshResultBuilder.setMultiFaceLandmarks(
                getProtoVector(packets.get(LANDMARKS_INDEX), NormalizedLandmarkList.parser()));
          } catch (MediaPipeException e) {
            reportError("Error occurs when getting MediaPipe facemesh landmarks.", e);
          }
          return faceMeshResultBuilder
              .setImagePacket(packets.get(INPUT_IMAGE_INDEX))
              .setTimestamp(
                  staticImageMode ? Long.MIN_VALUE : packets.get(INPUT_IMAGE_INDEX).getTimestamp())
              .build();
        });

    SolutionInfo solutionInfo =
        SolutionInfo.builder()
            .setBinaryGraphPath(options.runOnGpu() ? GPU_GRAPH_NAME : CPU_GRAPH_NAME)
            .setImageInputStreamName(IMAGE_INPUT_STREAM)
            .setOutputStreamNames(OUTPUT_STREAMS)
            .setStaticImageMode(options.staticImageMode())
            .build();

    initialize(context, solutionInfo, outputHandler);
    Map<String, Packet> inputSidePackets = new HashMap<>();
    inputSidePackets.put(NUM_FACES, packetCreator.createInt32(options.maxNumFaces()));
    inputSidePackets.put(WITH_ATTENTION, packetCreator.createBool(options.refineLandmarks()));
    inputSidePackets.put(USE_PREV_LANDMARKS, packetCreator.createBool(!options.staticImageMode()));
    start(inputSidePackets);
  }

  /**
   * Sets a callback to be invoked when a {@link FaceMeshResult} becomes available.
   *
   * @param listener the {@link ResultListener} callback.
   */
  public void setResultListener(ResultListener<FaceMeshResult> listener) {
    this.outputHandler.setResultListener(listener);
  }

  /**
   * Sets a callback to be invoked when the Face Mesh solution throws errors.
   *
   * @param listener the {@link ErrorListener} callback.
   */
  public void setErrorListener(@Nullable ErrorListener listener) {
    this.outputHandler.setErrorListener(listener);
    this.errorListener = listener;
  }
}
