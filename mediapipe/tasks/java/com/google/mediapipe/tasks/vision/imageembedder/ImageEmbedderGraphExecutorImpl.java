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

package com.google.mediapipe.tasks.vision.imageembedder;

import android.content.Context;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.EmbeddingResult;
import com.google.mediapipe.tasks.components.containers.proto.EmbeddingsProto;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.vision.core.BaseVisionTaskApi;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imageembedder.ImageEmbedder.ImageEmbedderOptions;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Graph-based executor implementation for standard MediaPipe Image Embedder execution. */
final class ImageEmbedderGraphExecutorImpl extends BaseVisionTaskApi
    implements ImageEmbedderExecutor {
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("EMBEDDINGS:embeddings_out", "IMAGE:image_out"));
  private static final int EMBEDDINGS_OUT_STREAM_INDEX = 0;
  private static final int IMAGE_OUT_STREAM_INDEX = 1;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph";

  @SuppressWarnings("EnumOrdinal")
  public static ImageEmbedderGraphExecutorImpl create(
      Context context, ImageEmbedderOptions options) {
    OutputHandler<ImageEmbedderResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<ImageEmbedderResult, MPImage>() {
          @Override
          public ImageEmbedderResult convertToTaskResult(List<Packet> packets) {
            try {
              return ImageEmbedderResult.create(
                  EmbeddingResult.createFromProto(
                      PacketGetter.getProto(
                          packets.get(EMBEDDINGS_OUT_STREAM_INDEX),
                          EmbeddingsProto.EmbeddingResult.getDefaultInstance())),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      options.runningMode(), packets.get(EMBEDDINGS_OUT_STREAM_INDEX)));
            } catch (IOException e) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INTERNAL.ordinal(), e.getMessage());
            }
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmap(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    options.resultListener().ifPresent(handler::setResultListener);
    options.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<ImageEmbedderOptions>builder()
                .setTaskName(ImageEmbedder.class.getSimpleName())
                .setTaskRunningModeName(options.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(options.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new ImageEmbedderGraphExecutorImpl(runner, options.runningMode());
  }

  private ImageEmbedderGraphExecutorImpl(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  @Override
  public ImageEmbedderResult embed(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    return (ImageEmbedderResult) processImageData(image, imageProcessingOptions);
  }

  @Override
  public ImageEmbedderResult embedForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    return (ImageEmbedderResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  @Override
  public void embedAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    sendLiveStreamData(image, imageProcessingOptions, timestampMs);
  }
}
