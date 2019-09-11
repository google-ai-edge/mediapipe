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

#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

// This test shows the default behavior of DefaultInputStreamHandler when
// batching is disabled.
TEST(DefaultInputStreamHandlerTest, NoBatchingWorks) {
  // A single calculator with two input streams, and two output streams. This
  // calculator passes all the input packets along.
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input0"
        input_stream: "input1"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input0"
          input_stream: "input1"
          output_stream: "output0"
          output_stream: "output1"
          input_stream_handler {
            input_stream_handler: "DefaultInputStreamHandler"
            options: {
              [mediapipe.DefaultInputStreamHandlerOptions.ext]: {
                batch_size: 1
              }
            }
          }
        })");
  std::vector<Packet> sink_0, sink_1;
  tool::AddVectorSink("output0", &config, &sink_0);
  tool::AddVectorSink("output1", &config, &sink_1);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(1)).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // No packets expected as the second stream is not ready to be processed.
  EXPECT_EQ(0, sink_0.size());
  EXPECT_EQ(0, sink_1.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", Adopt(new int(2)).At(Timestamp(2))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // First stream can produce output because the timestamp bound of the second
  // stream is higher.
  EXPECT_EQ(1, sink_0.size());
  EXPECT_EQ(0, sink_1.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(2)).At(Timestamp(2))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // Both streams have packets at the same timestamp, therefore both can produce
  // packets.
  EXPECT_EQ(2, sink_0.size());
  EXPECT_EQ(1, sink_1.size());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// This test shows the effect of batching on the DefaultInputStreamHandler.
TEST(DefaultInputStreamHandlerTest, Batches) {
  // A single batching calculator with one input stream and one output stream.
  // This calculator passes all the input packets onto the output streams.
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input0"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input0"
          output_stream: "output0"
          input_stream_handler {
            input_stream_handler: "DefaultInputStreamHandler"
            options: {
              [mediapipe.DefaultInputStreamHandlerOptions.ext]: {
                batch_size: 2
              }
            }
          }
        })");
  std::vector<Packet> sink;
  tool::AddVectorSink("output0", &config, &sink);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(1)).At(Timestamp(1))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  // There shouldn't be any outputs until a set of two packets is batched.
  EXPECT_TRUE(sink.empty());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(2)).At(Timestamp(2))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  // There should be two packets, processed during a single invocation.
  ASSERT_EQ(2, sink.size());
  EXPECT_THAT(std::vector<int>({sink[0].Get<int>(), sink[1].Get<int>()}),
              testing::ElementsAre(1, 2));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(3)).At(Timestamp(3))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  // There shouldn't be any outputs until another set of two packets is batched.
  EXPECT_EQ(2, sink.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(4)).At(Timestamp(4))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // The new batch was complete. There should be two more output packets.
  ASSERT_EQ(4, sink.size());
  EXPECT_THAT(std::vector<int>({sink[0].Get<int>(), sink[1].Get<int>(),
                                sink[2].Get<int>(), sink[3].Get<int>()}),
              testing::ElementsAre(1, 2, 3, 4));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// This test shows that any packets get flushed (outputted) when the input
// streams are closed.
TEST(DefaultInputStreamHandlerTest, BatchIsFlushedWhenClosing) {
  // A single batching calculator with one input stream and one output stream.
  // This calculator passes all the input packets onto the output streams.
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input0"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input0"
          output_stream: "output0"
          input_stream_handler {
            input_stream_handler: "DefaultInputStreamHandler"
            options: {
              [mediapipe.DefaultInputStreamHandlerOptions.ext]: {
                batch_size: 2
              }
            }
          }
        })");
  std::vector<Packet> sink;
  tool::AddVectorSink("output0", &config, &sink);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(1)).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // There shouldn't be any outputs until a set of two packets is batched.
  EXPECT_TRUE(sink.empty());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(2)).At(Timestamp(2))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // There should be two packets, processed during a single invocation.
  ASSERT_EQ(2, sink.size());
  EXPECT_THAT(std::vector<int>({sink[0].Get<int>(), sink[1].Get<int>()}),
              testing::ElementsAre(1, 2));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(3)).At(Timestamp(3))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  // There shouldn't be any outputs until another set of two packets is batched.
  EXPECT_EQ(2, sink.size());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // When the streams are done, the packets currently being batched should be
  // flushed out.
  ASSERT_EQ(3, sink.size());
  // Batched outputs should be in correct order.
  EXPECT_THAT(std::vector<int>(
                  {sink[0].Get<int>(), sink[1].Get<int>(), sink[2].Get<int>()}),
              testing::ElementsAre(1, 2, 3));
}

// This test shows that calculators won't propagate timestamp while they are
// batching except for the first timestamp of the batch.
TEST(DefaultInputStreamHandlerTest, DoesntPropagateTimestampWhenBatching) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input0"
        input_stream: "input1"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input1"
          output_stream: "input1_batched"
          input_stream_handler {
            input_stream_handler: "DefaultInputStreamHandler"
            options: {
              [mediapipe.DefaultInputStreamHandlerOptions.ext]: {
                batch_size: 3
              }
            }
          }
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input0"
          input_stream: "input1_batched"
          output_stream: "output"
          output_stream: "dummy"
        })");
  std::vector<Packet> sink;
  tool::AddVectorSink("output", &config, &sink);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(0)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_TRUE(sink.empty());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(1)).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", Adopt(new int(1)).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // Both calculators have packet 1. First node is currently batching and it
  // propagates the first input timestamp in the batch. Therefore, the
  // second node should produce output for the packet at 0.
  EXPECT_EQ(1, sink.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(2)).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", Adopt(new int(2)).At(Timestamp(2))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // Due to batching on the first node, timestamp is not propagated for the
  // packet at timestamp 2. Therefore, the second node cannot process the packet
  // at timestamp 1.
  EXPECT_EQ(1, sink.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", Adopt(new int(3)).At(Timestamp(3))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", Adopt(new int(3)).At(Timestamp(3))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // Batching is complete on the first node. It produced outputs at timestamp 1,
  // 2, and 3. The first node can now process the input packets at timestamps 1,
  // 2, and 3 as well.
  EXPECT_EQ(4, sink.size());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(4, sink.size());
}

}  // namespace
}  // namespace mediapipe
