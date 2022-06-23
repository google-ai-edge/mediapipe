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
"""MediaPipe SolutionBase module.

MediaPipe SolutionBase is the common base class for the high-level MediaPipe
Solution APIs such as BlazeFace, hand tracking, and BlazePose. The SolutionBase
class contains the shared logic among the high-level Solution APIs including
graph initialization, processing image/audio data, and graph shutdown. Thus,
users can easily create new MediaPipe Solution APIs on top of the SolutionBase
class.
"""

import collections
import enum
import os
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Union

import numpy as np

from google.protobuf.internal import containers
from google.protobuf import descriptor
from google.protobuf import message
# resources dependency
# pylint: disable=unused-import
# pylint: enable=unused-import
from mediapipe.framework import calculator_pb2
# pylint: disable=unused-import
from mediapipe.framework.formats import detection_pb2
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.image import image_transformation_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.util import landmarks_smoothing_calculator_pb2
from mediapipe.calculators.util import logic_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import rect_pb2
from mediapipe.modules.objectron.calculators import annotation_data_pb2
from mediapipe.modules.objectron.calculators import lift_2d_frame_annotation_to_3d_calculator_pb2
# pylint: enable=unused-import
from mediapipe.python._framework_bindings import calculator_graph
from mediapipe.python._framework_bindings import image_frame
from mediapipe.python._framework_bindings import packet
from mediapipe.python._framework_bindings import resource_util
from mediapipe.python._framework_bindings import validated_graph_config
import mediapipe.python.packet_creator as packet_creator
import mediapipe.python.packet_getter as packet_getter

RGB_CHANNELS = 3
# TODO: Enable calculator options modification for more calculators.
CALCULATOR_TO_OPTIONS = {
    'ConstantSidePacketCalculator':
        constant_side_packet_calculator_pb2.ConstantSidePacketCalculatorOptions,
    'ImageTransformationCalculator':
        image_transformation_calculator_pb2
        .ImageTransformationCalculatorOptions,
    'LandmarksSmoothingCalculator':
        landmarks_smoothing_calculator_pb2.LandmarksSmoothingCalculatorOptions,
    'LogicCalculator':
        logic_calculator_pb2.LogicCalculatorOptions,
    'ThresholdingCalculator':
        thresholding_calculator_pb2.ThresholdingCalculatorOptions,
    'TensorsToDetectionsCalculator':
        tensors_to_detections_calculator_pb2
        .TensorsToDetectionsCalculatorOptions,
    'Lift2DFrameAnnotationTo3DCalculator':
        lift_2d_frame_annotation_to_3d_calculator_pb2
        .Lift2DFrameAnnotationTo3DCalculatorOptions,
}


def type_names_from_oneof(oneof_type_name: str) -> Optional[List[str]]:
  if oneof_type_name.startswith('OneOf<') and oneof_type_name.endswith('>'):
    comma_separated_types = oneof_type_name[len('OneOf<'):-len('>')]
    return [n.strip() for n in comma_separated_types.split(',')]
  return None


# TODO: Support more packet data types, such as "Any" type.
@enum.unique
class PacketDataType(enum.Enum):
  """The packet data types supported by the SolutionBase class."""
  STRING = 'string'
  BOOL = 'bool'
  BOOL_LIST = 'bool_list'
  INT = 'int'
  INT_LIST = 'int_list'
  FLOAT = 'float'
  FLOAT_LIST = 'float_list'
  AUDIO = 'matrix'
  IMAGE = 'image'
  IMAGE_FRAME = 'image_frame'
  PROTO = 'proto'
  PROTO_LIST = 'proto_list'

  @staticmethod
  def from_registered_name(registered_name: str) -> 'PacketDataType':
    try:
      return NAME_TO_TYPE[registered_name]
    except KeyError as e:
      names = type_names_from_oneof(registered_name)
      if names:
        for n in names:
          if n in NAME_TO_TYPE.keys():
            return NAME_TO_TYPE[n]
      raise e

NAME_TO_TYPE: Mapping[str, 'PacketDataType'] = {
    'string':
        PacketDataType.STRING,
    'bool':
        PacketDataType.BOOL,
    '::std::vector<bool>':
        PacketDataType.BOOL_LIST,
    'int':
        PacketDataType.INT,
    '::std::vector<int>':
        PacketDataType.INT_LIST,
    'int64':
        PacketDataType.INT,
    '::std::vector<int64>':
        PacketDataType.INT_LIST,
    'float':
        PacketDataType.FLOAT,
    '::std::vector<float>':
        PacketDataType.FLOAT_LIST,
    '::mediapipe::Matrix':
        PacketDataType.AUDIO,
    '::mediapipe::ImageFrame':
        PacketDataType.IMAGE_FRAME,
    '::mediapipe::Classification':
        PacketDataType.PROTO,
    '::mediapipe::ClassificationList':
        PacketDataType.PROTO,
    '::mediapipe::ClassificationListCollection':
        PacketDataType.PROTO,
    '::mediapipe::Detection':
        PacketDataType.PROTO,
    '::mediapipe::DetectionList':
        PacketDataType.PROTO,
    '::mediapipe::Landmark':
        PacketDataType.PROTO,
    '::mediapipe::LandmarkList':
        PacketDataType.PROTO,
    '::mediapipe::LandmarkListCollection':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmark':
        PacketDataType.PROTO,
    '::mediapipe::FrameAnnotation':
        PacketDataType.PROTO,
    '::mediapipe::Trigger':
        PacketDataType.PROTO,
    '::mediapipe::Rect':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedRect':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmarkList':
        PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmarkListCollection':
        PacketDataType.PROTO,
    '::mediapipe::Image':
        PacketDataType.IMAGE,
    '::std::vector<::mediapipe::Classification>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::ClassificationList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Detection>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::DetectionList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Landmark>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::LandmarkList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedLandmark>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedLandmarkList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Rect>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedRect>':
        PacketDataType.PROTO_LIST,
}


class SolutionBase:
  """The common base class for the high-level MediaPipe Solution APIs.

  The SolutionBase class contains the shared logic among the high-level solution
  APIs including graph initialization, processing image/audio data, and graph
  shutdown.

  Example usage:
    with solution_base.SolutionBase(
        binary_graph_path='mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb',
        side_inputs={'num_hands': 2}) as hand_tracker:
      # Read an image and convert the BGR image to RGB.
      input_image = cv2.cvtColor(cv2.imread('/tmp/hand.png'), COLOR_BGR2RGB)
      results = hand_tracker.process(input_image)
      print(results.palm_detections)
      print(results.multi_hand_landmarks)
  """

  def __init__(
      self,
      binary_graph_path: Optional[str] = None,
      graph_config: Optional[calculator_pb2.CalculatorGraphConfig] = None,
      calculator_params: Optional[Mapping[str, Any]] = None,
      graph_options: Optional[message.Message] = None,
      side_inputs: Optional[Mapping[str, Any]] = None,
      outputs: Optional[List[str]] = None,
      stream_type_hints: Optional[Mapping[str, PacketDataType]] = None):
    """Initializes the SolutionBase object.

    Args:
      binary_graph_path: The path to a binary mediapipe graph file (.binarypb).
      graph_config: A CalculatorGraphConfig proto message or its text proto
        format.
      calculator_params: A mapping from the
        {calculator_name}.{options_field_name} str to the field value.
      graph_options: The graph options protobuf for the mediapipe graph.
      side_inputs: A mapping from the side packet name to the packet raw data.
      outputs: A list of the graph output stream names to observe. If the list
        is empty, all the output streams listed in the graph config will be
        automatically observed by default.
      stream_type_hints: A mapping from the stream name to its packet type hint.

    Raises:
      FileNotFoundError: If the binary graph file can't be found.
      RuntimeError: If the underlying calculator graph can't be successfully
        initialized or started.
      ValueError: If any of the following:
        a) If not exactly one of 'binary_graph_path' or 'graph_config' arguments
        is provided.
        b) If the graph validation process contains error.
        c) If the registered type name of the streams and side packets can't be
        found.
        d) If the calculator options of the calculator listed in
        calculator_params is not allowed to be modified.
        e) If the calculator options field is a repeated field but the field
        value to be set is not iterable.
        f) If not all calculator params are valid.
    """
    if bool(binary_graph_path) == bool(graph_config):
      raise ValueError(
          "Must provide exactly one of 'binary_graph_path' or 'graph_config'.")
    # MediaPipe package root path
    root_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    resource_util.set_resource_dir(root_path)
    validated_graph = validated_graph_config.ValidatedGraphConfig()
    if binary_graph_path:
      validated_graph.initialize(
          binary_graph_path=os.path.join(root_path, binary_graph_path))
    else:
      validated_graph.initialize(graph_config=graph_config)

    canonical_graph_config_proto = self._initialize_graph_interface(
        validated_graph, side_inputs, outputs, stream_type_hints)
    if calculator_params:
      self._modify_calculator_options(canonical_graph_config_proto,
                                      calculator_params)
    if graph_options:
      self._set_extension(canonical_graph_config_proto.graph_options,
                          graph_options)

    self._graph = calculator_graph.CalculatorGraph(
        graph_config=canonical_graph_config_proto)
    self._simulated_timestamp = 0
    self._graph_outputs = {}

    def callback(stream_name: str, output_packet: packet.Packet) -> None:
      self._graph_outputs[stream_name] = output_packet

    for stream_name in self._output_stream_type_info.keys():
      self._graph.observe_output_stream(stream_name, callback, True)

    self._input_side_packets = {
        name: self._make_packet(self._side_input_type_info[name], data)
        for name, data in (side_inputs or {}).items()
    }
    self._graph.start_run(self._input_side_packets)

  # TODO: Use "inspect.Parameter" to fetch the input argument names and
  # types from "_input_stream_type_info" and then auto generate the process
  # method signature by "inspect.Signature" in __init__.
  def process(
      self, input_data: Union[np.ndarray, Mapping[str, Union[np.ndarray,
                                                             message.Message]]]
  ) -> NamedTuple:
    """Processes a set of RGB image data and output SolutionOutputs.

    Args:
      input_data: Either a single numpy ndarray object representing the solo
        image input of a graph or a mapping from the stream name to the image or
        proto data that represents every input streams of a graph.

    Raises:
      NotImplementedError: If input_data contains audio data or a list of proto
        objects.
      RuntimeError: If the underlying graph occurs any error.
      ValueError: If the input image data is not three channel RGB.

    Returns:
      A NamedTuple object that contains the output data of a graph run.
        The field names in the NamedTuple object are mapping to the graph output
        stream names.

    Examples:
      solution = solution_base.SolutionBase(graph_config=hand_landmark_graph)
      results = solution.process(cv2.imread('/tmp/hand0.png')[:, :, ::-1])
      print(results.detection)
      results = solution.process(
          {'video_in' : cv2.imread('/tmp/hand1.png')[:, :, ::-1]})
      print(results.hand_landmarks)
    """
    self._graph_outputs.clear()

    if isinstance(input_data, np.ndarray):
      if len(self._input_stream_type_info.keys()) != 1:
        raise ValueError(
            "Can't process single image input since the graph has more than one input streams."
        )
      input_dict = {next(iter(self._input_stream_type_info)): input_data}
    else:
      input_dict = input_data

    # Set the timestamp increment to 33333 us to simulate the 30 fps video
    # input.
    self._simulated_timestamp += 33333
    for stream_name, data in input_dict.items():
      input_stream_type = self._input_stream_type_info[stream_name]
      if (input_stream_type == PacketDataType.PROTO_LIST or
          input_stream_type == PacketDataType.AUDIO):
        # TODO: Support audio data.
        raise NotImplementedError(
            f'SolutionBase can only process non-audio and non-proto-list data. '
            f'{self._input_stream_type_info[stream_name].name} '
            f'type is not supported yet.')
      elif (input_stream_type == PacketDataType.IMAGE_FRAME or
            input_stream_type == PacketDataType.IMAGE):
        if data.shape[2] != RGB_CHANNELS:
          raise ValueError('Input image must contain three channel rgb data.')
        self._graph.add_packet_to_input_stream(
            stream=stream_name,
            packet=self._make_packet(input_stream_type,
                                     data).at(self._simulated_timestamp))
      else:
        self._graph.add_packet_to_input_stream(
            stream=stream_name,
            packet=self._make_packet(input_stream_type,
                                     data).at(self._simulated_timestamp))

    self._graph.wait_until_idle()
    # Create a NamedTuple object where the field names are mapping to the graph
    # output stream names.
    solution_outputs = collections.namedtuple(
        'SolutionOutputs', self._output_stream_type_info.keys())
    for stream_name in self._output_stream_type_info.keys():
      if stream_name in self._graph_outputs:
        setattr(
            solution_outputs, stream_name,
            self._get_packet_content(self._output_stream_type_info[stream_name],
                                     self._graph_outputs[stream_name]))
      else:
        setattr(solution_outputs, stream_name, None)

    return solution_outputs

  def close(self) -> None:
    """Closes all the input sources and the graph."""
    self._graph.close()
    self._graph = None
    self._input_stream_type_info = None
    self._output_stream_type_info = None

  def reset(self) -> None:
    """Resets the graph for another run."""
    if self._graph:
      self._graph.close()
      self._graph.start_run(self._input_side_packets)

  def _initialize_graph_interface(
      self,
      validated_graph: validated_graph_config.ValidatedGraphConfig,
      side_inputs: Optional[Mapping[str, Any]] = None,
      outputs: Optional[List[str]] = None,
      stream_type_hints: Optional[Mapping[str, PacketDataType]] = None):
    """Gets graph interface type information and returns the canonical graph config proto."""

    canonical_graph_config_proto = calculator_pb2.CalculatorGraphConfig()
    canonical_graph_config_proto.ParseFromString(validated_graph.binary_config)

    # Gets name from a 'TAG:index:name' str.
    def get_name(tag_index_name):
      return tag_index_name.split(':')[-1]

    # Gets the packet type information of the input streams and output streams
    # from the user provided stream_type_hints field or validated calculator
    # graph. The mappings from the stream names to the packet data types is
    # for deciding which packet creator and getter methods to call in the
    # process() method.
    def get_stream_packet_type(packet_tag_index_name):
      stream_name = get_name(packet_tag_index_name)
      if stream_type_hints and stream_name in stream_type_hints.keys():
        return stream_type_hints[stream_name]
      return PacketDataType.from_registered_name(
          validated_graph.registered_stream_type_name(stream_name))

    self._input_stream_type_info = {
        get_name(tag_index_name): get_stream_packet_type(tag_index_name)
        for tag_index_name in canonical_graph_config_proto.input_stream
    }

    if not outputs:
      output_streams = canonical_graph_config_proto.output_stream
    else:
      output_streams = outputs
    self._output_stream_type_info = {
        get_name(tag_index_name): get_stream_packet_type(tag_index_name)
        for tag_index_name in output_streams
    }

    # Gets the packet type information of the input side packets from the
    # validated calculator graph. The mappings from the side packet names to the
    # packet data types is for making the input_side_packets dict for graph
    # start_run().
    def get_side_packet_type(packet_tag_index_name):
      return PacketDataType.from_registered_name(
          validated_graph.registered_side_packet_type_name(
              get_name(packet_tag_index_name)))

    self._side_input_type_info = {
        get_name(tag_index_name): get_side_packet_type(tag_index_name)
        for tag_index_name, _ in (side_inputs or {}).items()
    }
    return canonical_graph_config_proto

  def _modify_calculator_options(
      self, calculator_graph_config: calculator_pb2.CalculatorGraphConfig,
      calculator_params: Mapping[str, Any]) -> None:
    """Modifies the CalculatorOptions of the calculators listed in calculator_params."""

    # Reorganizes the calculator options field data by calculator name and puts
    # all the field data of the same calculator in a list.
    def generate_nested_calculator_params(flat_map):
      nested_map = {}
      for compound_name, field_value in flat_map.items():
        calculator_and_field_name = compound_name.split('.')
        if len(calculator_and_field_name) != 2:
          raise ValueError(
              f'The key "{compound_name}" in the calculator_params is invalid.')
        calculator_name = calculator_and_field_name[0]
        field_name = calculator_and_field_name[1]
        if calculator_name in nested_map:
          nested_map[calculator_name].append((field_name, field_value))
        else:
          nested_map[calculator_name] = [(field_name, field_value)]
      return nested_map

    def modify_options_fields(calculator_options, options_field_list):
      for field_name, field_value in options_field_list:
        if field_value is None:
          calculator_options.ClearField(field_name)
        else:
          field_label = calculator_options.DESCRIPTOR.fields_by_name[
              field_name].label
          if field_label == descriptor.FieldDescriptor.LABEL_REPEATED:
            if not isinstance(field_value, Iterable):
              raise ValueError(
                  f'{field_name} is a repeated proto field but the value '
                  f'to be set is {type(field_value)}, which is not iterable.')
            # TODO: Support resetting the entire repeated field
            # (array-option) and changing the individual values in the repeated
            # field (array-element-option).
            calculator_options.ClearField(field_name)
            for elem in field_value:
              getattr(calculator_options, field_name).append(elem)
          else:
            setattr(calculator_options, field_name, field_value)

    nested_calculator_params = generate_nested_calculator_params(
        calculator_params)

    num_modified = 0
    for node in calculator_graph_config.node:
      if node.name not in nested_calculator_params:
        continue
      options_type = CALCULATOR_TO_OPTIONS.get(node.calculator)
      if options_type is None:
        raise ValueError(
            f'Modifying the calculator options of {node.name} is not supported.'
        )
      options_field_list = nested_calculator_params[node.name]
      if node.HasField('options') and node.node_options:
        raise ValueError(
            f'Cannot modify the calculator options of {node.name} because it '
            f'has both options and node_options fields.')
      if node.node_options:
        # The "node_options" case for the proto3 syntax.
        node_options_modified = False
        for elem in node.node_options:
          type_name = elem.type_url.split('/')[-1]
          if type_name == options_type.DESCRIPTOR.full_name:
            calculator_options = options_type.FromString(elem.value)
            modify_options_fields(calculator_options, options_field_list)
            elem.value = calculator_options.SerializeToString()
            node_options_modified = True
            break
        # There is no existing node_options being modified. Add a new
        # node_options instead.
        if not node_options_modified:
          calculator_options = options_type()
          modify_options_fields(calculator_options, options_field_list)
          node.node_options.add().Pack(calculator_options)
      else:
        # The "options" case for the proto2 syntax as well as the fallback
        # when the calculator doesn't have either "options" or "node_options".
        modify_options_fields(node.options.Extensions[options_type.ext],
                              options_field_list)

      num_modified += 1
      # Exits the loop early when every elements in nested_calculator_params
      # have been visited.
      if num_modified == len(nested_calculator_params):
        break
    if num_modified < len(nested_calculator_params):
      raise ValueError('Not all calculator params are valid.')

  def create_graph_options(self, options_message: message.Message,
                           values: Mapping[str, Any]) -> message.Message:
    """Sets protobuf field values.

    Args:
      options_message: the options protobuf message.
      values: field value pairs, where each field may be a "." separated path.

    Returns:
      the options protobuf message.
    """

    if hasattr(values, 'items'):
      values = values.items()
    for pair in values:
      (field, value) = pair
      fields = field.split('.')
      m = options_message
      while len(fields) > 1:
        m = getattr(m, fields[0])
        del fields[0]
      v = getattr(m, fields[0])
      if hasattr(v, 'append'):
        del v[:]
        v.extend(value)
      elif hasattr(v, 'CopyFrom'):
        v.CopyFrom(value)
      else:
        setattr(m, fields[0], value)
    return options_message

  def _set_extension(self,
                     extension_list: containers.RepeatedCompositeFieldContainer,
                     extension_value: message.Message) -> None:
    """Sets one value in a repeated protobuf.Any extension field."""
    for extension_any in extension_list:
      if extension_any.Is(extension_value.DESCRIPTOR):
        v = type(extension_value)()
        extension_any.Unpack(v)
        v.MergeFrom(extension_value)
        extension_any.Pack(v)
        return
    extension_list.add().Pack(extension_value)

  def _make_packet(self, packet_data_type: PacketDataType,
                   data: Any) -> packet.Packet:
    if (packet_data_type == PacketDataType.IMAGE_FRAME or
        packet_data_type == PacketDataType.IMAGE):
      return getattr(packet_creator, 'create_' + packet_data_type.value)(
          data, image_format=image_frame.ImageFormat.SRGB)
    else:
      return getattr(packet_creator, 'create_' + packet_data_type.value)(data)

  def _get_packet_content(self, packet_data_type: PacketDataType,
                          output_packet: packet.Packet) -> Any:
    """Gets packet content from a packet by type.

    Args:
      packet_data_type: The supported packet data type.
      output_packet: The packet to get content from.

    Returns:
      Packet content by packet data type. None to indicate "no output".

    """

    if output_packet.is_empty():
      return None
    if packet_data_type == PacketDataType.STRING:
      return packet_getter.get_str(output_packet)
    elif (packet_data_type == PacketDataType.IMAGE_FRAME or
          packet_data_type == PacketDataType.IMAGE):
      return getattr(packet_getter, 'get_' +
                     packet_data_type.value)(output_packet).numpy_view()
    else:
      return getattr(packet_getter, 'get_' + packet_data_type.value)(
          output_packet)

  def __enter__(self):
    """A "with" statement support."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Closes all the input sources and the graph."""
    self.close()
