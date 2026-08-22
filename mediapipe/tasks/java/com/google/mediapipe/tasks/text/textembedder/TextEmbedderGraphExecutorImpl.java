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

package com.google.mediapipe.tasks.text.textembedder;

import android.content.Context;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.tasks.components.containers.EmbeddingResult;
import com.google.mediapipe.tasks.components.containers.proto.EmbeddingsProto;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder.TextEmbedderOptions;
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder.TextFormatContext;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Graph-based executor implementation for standard MediaPipe Text Embedder execution. */
final class TextEmbedderGraphExecutorImpl implements TextEmbedderExecutor {
  private static final String INPUT_STREAM_NAME = "text_in";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("TEXT:" + INPUT_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("EMBEDDINGS:embeddings_out"));

  private static final int EMBEDDINGS_OUT_STREAM_INDEX = 0;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.text.text_embedder.TextEmbedderGraph";

  private final TaskRunner runner;

  public static TextEmbedderGraphExecutorImpl create(Context context, TextEmbedderOptions options) {
    OutputHandler<TextEmbedderResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<TextEmbedderResult, Void>() {
          @Override
          public TextEmbedderResult convertToTaskResult(List<Packet> packets) {
            try {
              return TextEmbedderResult.create(
                  EmbeddingResult.createFromProto(
                      PacketGetter.getProto(
                          packets.get(EMBEDDINGS_OUT_STREAM_INDEX),
                          EmbeddingsProto.EmbeddingResult.getDefaultInstance())),
                  packets.get(EMBEDDINGS_OUT_STREAM_INDEX).getTimestamp());
            } catch (IOException e) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INTERNAL, e.getMessage(), e);
            }
          }

          @Override
          public Void convertToTaskInput(List<Packet> packets) {
            return null;
          }
        });
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<TextEmbedderOptions>builder()
                .setTaskName(TextEmbedder.class.getSimpleName())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new TextEmbedderGraphExecutorImpl(runner);
  }

  private TextEmbedderGraphExecutorImpl(TaskRunner runner) {
    this.runner = runner;
  }

  @Override
  public TextEmbedderResult embed(String inputText) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(INPUT_STREAM_NAME, runner.getPacketCreator().createString(inputText));
    return (TextEmbedderResult) runner.process(inputPackets);
  }

  @Override
  public TextEmbedderResult embed(String inputText, TextFormatContext formatContext) {
    Map<String, Packet> inputPackets = new HashMap<>();
    String processedText = TextEmbedder.getGeckoEmbeddingText(inputText, formatContext);
    inputPackets.put(INPUT_STREAM_NAME, runner.getPacketCreator().createString(processedText));
    return (TextEmbedderResult) runner.process(inputPackets);
  }

  @Override
  public void close() {
    runner.close();
  }
}
