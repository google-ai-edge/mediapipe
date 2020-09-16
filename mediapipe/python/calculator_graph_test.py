# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for mediapipe.python._framework_bindings.calculator_graph."""

# Dependency imports

from absl.testing import absltest
import mediapipe as mp
from google.protobuf import text_format
from mediapipe.framework import calculator_pb2


class GraphTest(absltest.TestCase):

  def testInvalidBinaryGraphFile(self):
    with self.assertRaisesRegex(FileNotFoundError, 'No such file or directory'):
      mp.CalculatorGraph(binary_graph_path='/tmp/abc.binarypb')

  def testInvalidNodeConfig(self):
    text_config = """
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        input_stream: 'in'
        output_stream: 'out'
      }
    """
    config_proto = calculator_pb2.CalculatorGraphConfig()
    text_format.Parse(text_config, config_proto)
    with self.assertRaisesRegex(
        ValueError,
        'Input and output streams to PassThroughCalculator must use matching tags and indexes.'
    ):
      mp.CalculatorGraph(graph_config=config_proto)

  def testInvalidCalculatorType(self):
    text_config = """
      node {
        calculator: 'SomeUnknownCalculator'
        input_stream: 'in'
        output_stream: 'out'
      }
    """
    config_proto = calculator_pb2.CalculatorGraphConfig()
    text_format.Parse(text_config, config_proto)
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to find Calculator \"SomeUnknownCalculator\"'):
      mp.CalculatorGraph(graph_config=config_proto)

  def testGraphInitializedWithProtoConfig(self):
    text_config = """
      max_queue_size: 1
      input_stream: 'in'
      output_stream: 'out'
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        output_stream: 'out'
      }
    """
    config_proto = calculator_pb2.CalculatorGraphConfig()
    text_format.Parse(text_config, config_proto)
    graph = mp.CalculatorGraph(graph_config=config_proto)

    hello_world_packet = mp.packet_creator.create_string('hello world')
    out = []
    graph = mp.CalculatorGraph(graph_config=config_proto)
    graph.observe_output_stream('out', lambda _, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet, timestamp=0)
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet.at(1))
    graph.close()
    self.assertEqual(graph.graph_input_stream_add_mode,
                     mp.GraphInputStreamAddMode.WAIT_TILL_NOT_FULL)
    self.assertEqual(graph.max_queue_size, 1)
    self.assertFalse(graph.has_error())
    self.assertLen(out, 2)
    self.assertEqual(out[0].timestamp, 0)
    self.assertEqual(out[1].timestamp, 1)
    self.assertEqual(mp.packet_getter.get_str(out[0]), 'hello world')
    self.assertEqual(mp.packet_getter.get_str(out[1]), 'hello world')

  def testGraphInitializedWithTextConfig(self):
    text_config = """
      max_queue_size: 1
      input_stream: 'in'
      output_stream: 'out'
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        output_stream: 'out'
      }
    """

    hello_world_packet = mp.packet_creator.create_string('hello world')
    out = []
    graph = mp.CalculatorGraph(graph_config=text_config)
    graph.observe_output_stream('out', lambda _, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet.at(0))
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet, timestamp=1)
    graph.close()
    self.assertEqual(graph.graph_input_stream_add_mode,
                     mp.GraphInputStreamAddMode.WAIT_TILL_NOT_FULL)
    self.assertEqual(graph.max_queue_size, 1)
    self.assertFalse(graph.has_error())
    self.assertLen(out, 2)
    self.assertEqual(out[0].timestamp, 0)
    self.assertEqual(out[1].timestamp, 1)
    self.assertEqual(mp.packet_getter.get_str(out[0]), 'hello world')
    self.assertEqual(mp.packet_getter.get_str(out[1]), 'hello world')

  def testGraphValidationAndInitialization(self):
    text_config = """
      max_queue_size: 1
      input_stream: 'in'
      output_stream: 'out'
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        output_stream: 'out'
      }
    """

    hello_world_packet = mp.packet_creator.create_string('hello world')
    out = []
    validated_graph_config = mp.ValidatedGraphConfig()
    self.assertFalse(validated_graph_config.initialized())
    validated_graph_config.initialize(graph_config=text_config)
    self.assertTrue(validated_graph_config.initialized())

    graph = mp.CalculatorGraph(validated_graph_config=validated_graph_config)
    graph.observe_output_stream('out', lambda _, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet.at(0))
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet, timestamp=1)
    graph.close()
    self.assertEqual(graph.graph_input_stream_add_mode,
                     mp.GraphInputStreamAddMode.WAIT_TILL_NOT_FULL)
    self.assertEqual(graph.max_queue_size, 1)
    self.assertFalse(graph.has_error())
    self.assertLen(out, 2)
    self.assertEqual(out[0].timestamp, 0)
    self.assertEqual(out[1].timestamp, 1)
    self.assertEqual(mp.packet_getter.get_str(out[0]), 'hello world')
    self.assertEqual(mp.packet_getter.get_str(out[1]), 'hello world')

  def testInsertPacketsWithSameTimestamp(self):
    text_config = """
      max_queue_size: 1
      input_stream: 'in'
      output_stream: 'out'
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        output_stream: 'out'
      }
    """
    config_proto = calculator_pb2.CalculatorGraphConfig()
    text_format.Parse(text_config, config_proto)

    hello_world_packet = mp.packet_creator.create_string('hello world')
    out = []
    graph = mp.CalculatorGraph(graph_config=config_proto)
    graph.observe_output_stream('out', lambda _, packet: out.append(packet))
    graph.start_run()
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet.at(0))
    graph.wait_until_idle()
    graph.add_packet_to_input_stream(
        stream='in', packet=hello_world_packet.at(0))
    with self.assertRaisesRegex(
        ValueError, 'Current minimum expected timestamp is 1 but received 0.'):
      graph.wait_until_idle()

  def testSidePacketGraph(self):
    text_config = """
      node {
        calculator: 'StringToUint64Calculator'
        input_side_packet: "string"
        output_side_packet: "number"
      }
    """
    config_proto = calculator_pb2.CalculatorGraphConfig()
    text_format.Parse(text_config, config_proto)
    graph = mp.CalculatorGraph(graph_config=config_proto)
    graph.start_run(
        input_side_packets={'string': mp.packet_creator.create_string('42')})
    graph.wait_until_done()
    self.assertFalse(graph.has_error())
    self.assertEqual(
        mp.packet_getter.get_uint(graph.get_output_side_packet('number')), 42)


if __name__ == '__main__':
  absltest.main()
