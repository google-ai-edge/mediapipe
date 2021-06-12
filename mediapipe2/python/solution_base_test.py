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

"""Tests for mediapipe.python.solution_base."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.framework import calculator_pb2
from mediapipe.framework.formats import detection_pb2
from mediapipe.python import solution_base

CALCULATOR_OPTIONS_TEST_GRAPH_CONFIG = """
  input_stream: 'image_in'
  output_stream: 'image_out'
  node {
    name: 'ImageTransformation'
    calculator: 'ImageTransformationCalculator'
    input_stream: 'IMAGE:image_in'
    output_stream: 'IMAGE:image_out'
    options: {
      [mediapipe.ImageTransformationCalculatorOptions.ext] {
         output_width: 10
         output_height: 10
      }
    }
    node_options: {
      [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
         output_width: 10
         output_height: 10
      }
    }
  }
"""


class SolutionBaseTest(parameterized.TestCase):

  def test_invalid_initialization_arguments(self):
    with self.assertRaisesRegex(
        ValueError,
        'Must provide exactly one of \'binary_graph_path\' or \'graph_config\'.'
    ):
      solution_base.SolutionBase()
    with self.assertRaisesRegex(
        ValueError,
        'Must provide exactly one of \'binary_graph_path\' or \'graph_config\'.'
    ):
      solution_base.SolutionBase(
          graph_config=calculator_pb2.CalculatorGraphConfig(),
          binary_graph_path='/tmp/no_such.binarypb')

  @parameterized.named_parameters(('no_graph_input_output_stream', """
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        output_stream: 'out'
      }
      """, RuntimeError, 'does not have a corresponding output stream.'),
                                  ('calcualtor_io_mismatch', """
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        input_stream: 'in2'
        output_stream: 'out'
      }
      """, ValueError, 'must use matching tags and indexes.'),
                                  ('unkown_registered_stream_type_name', """
      input_stream: 'in'
      output_stream: 'out'
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in'
        output_stream: 'out'
      }
      """, RuntimeError, 'Unable to find the type for stream \"in\".'))
  def test_invalid_config(self, text_config, error_type, error_message):
    config_proto = text_format.Parse(text_config,
                                     calculator_pb2.CalculatorGraphConfig())
    with self.assertRaisesRegex(error_type, error_message):
      solution_base.SolutionBase(graph_config=config_proto)

  def test_invalid_input_data_type(self):
    text_config = """
      input_stream: 'input_detections'
      output_stream: 'output_detections'
      node {
        calculator: 'DetectionUniqueIdCalculator'
        input_stream: 'DETECTIONS:input_detections'
        output_stream: 'DETECTIONS:output_detections'
      }
    """
    config_proto = text_format.Parse(text_config,
                                     calculator_pb2.CalculatorGraphConfig())
    with solution_base.SolutionBase(graph_config=config_proto) as solution:
      detection = detection_pb2.Detection()
      text_format.Parse('score: 0.5', detection)
      with self.assertRaisesRegex(
          NotImplementedError,
          'SolutionBase can only process image data. PROTO_LIST type is not supported.'
      ):
        solution.process({'input_detections': detection})

  def test_invalid_input_image_data(self):
    text_config = """
      input_stream: 'image_in'
      output_stream: 'image_out'
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:image_in'
        output_stream: 'IMAGE:transformed_image_in'
      }
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:transformed_image_in'
        output_stream: 'IMAGE:image_out'
      }
    """
    config_proto = text_format.Parse(text_config,
                                     calculator_pb2.CalculatorGraphConfig())
    with solution_base.SolutionBase(graph_config=config_proto) as solution:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        solution.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  @parameterized.named_parameters(('graph_without_side_packets', """
      input_stream: 'image_in'
      output_stream: 'image_out'
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:image_in'
        output_stream: 'IMAGE:transformed_image_in'
      }
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:transformed_image_in'
        output_stream: 'IMAGE:image_out'
      }
      """, None), ('graph_with_side_packets', """
      input_stream: 'image_in'
      input_side_packet: 'allow_signal'
      input_side_packet: 'rotation_degrees'
      output_stream: 'image_out'
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:image_in'
        input_side_packet: 'ROTATION_DEGREES:rotation_degrees'
        output_stream: 'IMAGE:transformed_image_in'
      }
      node {
        calculator: 'GateCalculator'
        input_stream: 'transformed_image_in'
        input_side_packet: 'ALLOW:allow_signal'
        output_stream: 'image_out_to_transform'
      }
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:image_out_to_transform'
        input_side_packet: 'ROTATION_DEGREES:rotation_degrees'
        output_stream: 'IMAGE:image_out'
      }""", {
          'allow_signal': True,
          'rotation_degrees': 0
      }))
  def test_solution_process(self, text_config, side_inputs):
    self._process_and_verify(
        config_proto=text_format.Parse(text_config,
                                       calculator_pb2.CalculatorGraphConfig()),
        side_inputs=side_inputs)

  def test_invalid_calculator_options(self):
    text_config = """
      input_stream: 'image_in'
      output_stream: 'image_out'
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:image_in'
        output_stream: 'IMAGE:transformed_image_in'
      }
      node {
        name: 'SignalGate'
        calculator: 'GateCalculator'
        input_stream: 'transformed_image_in'
        input_side_packet: 'ALLOW:allow_signal'
        output_stream: 'image_out_to_transform'
      }
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:image_out_to_transform'
        output_stream: 'IMAGE:image_out'
      }
    """
    config_proto = text_format.Parse(text_config,
                                     calculator_pb2.CalculatorGraphConfig())
    with self.assertRaisesRegex(
        ValueError,
        'Modifying the calculator options of SignalGate is not supported.'):
      solution_base.SolutionBase(
          graph_config=config_proto,
          calculator_params={'SignalGate.invalid_field': 'I am invalid'})

  def test_calculator_has_both_options_and_node_options(self):
    config_proto = text_format.Parse(CALCULATOR_OPTIONS_TEST_GRAPH_CONFIG,
                                     calculator_pb2.CalculatorGraphConfig())
    with self.assertRaisesRegex(ValueError,
                                'has both options and node_options fields.'):
      solution_base.SolutionBase(
          graph_config=config_proto,
          calculator_params={
              'ImageTransformation.output_width': 0,
              'ImageTransformation.output_height': 0
          })

  def test_modifying_calculator_proto2_options(self):
    config_proto = text_format.Parse(CALCULATOR_OPTIONS_TEST_GRAPH_CONFIG,
                                     calculator_pb2.CalculatorGraphConfig())
    # To test proto2 options only, remove the proto3 node_options field from the
    # graph config.
    self.assertEqual('ImageTransformation', config_proto.node[0].name)
    config_proto.node[0].ClearField('node_options')
    self._process_and_verify(
        config_proto=config_proto,
        calculator_params={
            'ImageTransformation.output_width': 0,
            'ImageTransformation.output_height': 0
        })

  def test_modifying_calculator_proto3_node_options(self):
    config_proto = text_format.Parse(CALCULATOR_OPTIONS_TEST_GRAPH_CONFIG,
                                     calculator_pb2.CalculatorGraphConfig())
    # To test proto3 node options only, remove the proto2 options field from the
    # graph config.
    self.assertEqual('ImageTransformation', config_proto.node[0].name)
    config_proto.node[0].ClearField('options')
    self._process_and_verify(
        config_proto=config_proto,
        calculator_params={
            'ImageTransformation.output_width': 0,
            'ImageTransformation.output_height': 0
        })

  def test_adding_calculator_options(self):
    config_proto = text_format.Parse(CALCULATOR_OPTIONS_TEST_GRAPH_CONFIG,
                                     calculator_pb2.CalculatorGraphConfig())
    # To test a calculator with no options field, remove both proto2 options and
    # proto3 node_options fields from the graph config.
    self.assertEqual('ImageTransformation', config_proto.node[0].name)
    config_proto.node[0].ClearField('options')
    config_proto.node[0].ClearField('node_options')
    self._process_and_verify(
        config_proto=config_proto,
        calculator_params={
            'ImageTransformation.output_width': 0,
            'ImageTransformation.output_height': 0
        })

  def _process_and_verify(self,
                          config_proto,
                          side_inputs=None,
                          calculator_params=None):
    input_image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    with solution_base.SolutionBase(
        graph_config=config_proto,
        side_inputs=side_inputs,
        calculator_params=calculator_params) as solution:
      outputs = solution.process(input_image)
      outputs2 = solution.process({'image_in': input_image})
    self.assertTrue(np.array_equal(input_image, outputs.image_out))
    self.assertTrue(np.array_equal(input_image, outputs2.image_out))


if __name__ == '__main__':
  absltest.main()
