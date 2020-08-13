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
"""The public facing packet creator APIs."""

from typing import List, Union

import numpy as np

from google.protobuf import message
from mediapipe.python._framework_bindings import _packet_creator
from mediapipe.python._framework_bindings import image_frame
from mediapipe.python._framework_bindings import packet


create_string = _packet_creator.create_string
create_bool = _packet_creator.create_bool
create_int = _packet_creator.create_int
create_int8 = _packet_creator.create_int8
create_int16 = _packet_creator.create_int16
create_int32 = _packet_creator.create_int32
create_int64 = _packet_creator.create_int64
create_uint8 = _packet_creator.create_uint8
create_uint16 = _packet_creator.create_uint16
create_uint32 = _packet_creator.create_uint32
create_uint64 = _packet_creator.create_uint64
create_float = _packet_creator.create_float
create_double = _packet_creator.create_double
create_int_array = _packet_creator.create_int_array
create_float_array = _packet_creator.create_float_array
create_int_vector = _packet_creator.create_int_vector
create_float_vector = _packet_creator.create_float_vector
create_string_vector = _packet_creator.create_string_vector
create_packet_vector = _packet_creator.create_packet_vector
create_string_to_packet_map = _packet_creator.create_string_to_packet_map
create_matrix = _packet_creator.create_matrix


def create_image_frame(
    data: Union[image_frame.ImageFrame, np.ndarray],
    *,
    image_format: image_frame.ImageFormat = None) -> packet.Packet:
  """Create a MediaPipe ImageFrame packet.

  A MediaPipe ImageFrame packet can be created from either the raw pixel data
  represented as a numpy array with one of the uint8, uint16, and float data
  types or an existing MediaPipe ImageFrame object. The data will be realigned
  and copied into an ImageFrame object inside of the packet.

  Args:
    data: A MediaPipe ImageFrame object or the raw pixel data that is
      represnted as a numpy ndarray.
    image_format: One of the image_frame.ImageFormat enum types.

  Returns:
    A MediaPipe ImageFrame Packet.

  Raises:
    ValueError:
      i) When "data" is a numpy ndarray, "image_format" is not provided.
      ii) When "data" is an ImageFrame object, the "image_format" arg doesn't
        match the image format of the "data" ImageFrame object.
    TypeError: If "image format" doesn't match "data" array's data type.

  Examples:
    np_array = np.random.randint(255, size=(321, 123, 3), dtype=np.uint8)
    image_frame_packet = mp.packet_creator.create_image_frame(
        image_format=mp.ImageFormat.SRGB, data=np_array)

    image_frame = mp.ImageFrame(image_format=mp.ImageFormat.SRGB, data=np_array)
    image_frame_packet = mp.packet_creator.create_image_frame(image_frame)

  """
  if isinstance(data, image_frame.ImageFrame):
    if image_format is not None and data.image_format != image_format:
      raise ValueError(
          'The provided image_format doesn\'t match the one from the data arg.')
    # pylint:disable=protected-access
    return _packet_creator._create_image_frame_with_copy(data)
    # pylint:enable=protected-access
  else:
    if image_format is None:
      raise ValueError('Please provide \'image_format\' with \'data\'.')
    # pylint:disable=protected-access
    return _packet_creator._create_image_frame_with_copy(image_format, data)
    # pylint:enable=protected-access


def create_proto(proto_message: message.Message) -> packet.Packet:
  """Create a MediaPipe protobuf message packet.

  Args:
    proto_message: A Python protobuf message.

  Returns:
    A MediaPipe protobuf message Packet.

  Raises:
    RuntimeError: If the protobuf message type is not registered in MediaPipe.

  Examples:
    detection = detection_pb2.Detection()
    text_format.Parse('score: 0.5', detection)
    packet = mp.packet_creator.create_proto(detection)
    output_detection = mp.packet_getter.get_proto(packet)
  """
  # pylint:disable=protected-access
  return _packet_creator._create_proto(proto_message.DESCRIPTOR.full_name,
                                       proto_message.SerializeToString())
  # pylint:enable=protected-access


def create_proto_vector(message_list: List[message.Message]) -> packet.Packet:
  raise NotImplementedError('create_proto_vector is not implemented.')
