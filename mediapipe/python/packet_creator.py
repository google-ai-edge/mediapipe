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
import warnings

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


def create_image_frame(data: Union[image_frame.ImageFrame, np.ndarray],
                       *,
                       image_format: image_frame.ImageFormat = None,
                       copy: bool = None) -> packet.Packet:
  """Create a MediaPipe ImageFrame packet.

  A MediaPipe ImageFrame packet can be created from an existing MediaPipe
  ImageFrame object and the data will be realigned and copied into a new
  ImageFrame object inside of the packet.

  A MediaPipe ImageFrame packet can also be created from the raw pixel data
  represented as a numpy array with one of the uint8, uint16, and float data
  types. There are three data ownership modes depending on how the 'copy' arg
  is set.

  i) Default mode
  If copy is not set, mutable data is always copied while the immutable data
  is by reference.

  ii) Copy mode (safe)
  If copy is set to True, the data will be realigned and copied into an
  ImageFrame object inside of the packet regardless the immutablity of the
  original data.

  iii) Reference mode (dangerous)
  If copy is set to False, the data will be forced to be shared. If the data is
  mutable (data.flags.writeable is True), a warning will be raised.

  Args:
    data: A MediaPipe ImageFrame object or the raw pixel data that is
      represnted as a numpy ndarray.
    image_format: One of the image_frame.ImageFormat enum types.
    copy: Indicate if the packet should copy the data from the numpy nparray.

  Returns:
    A MediaPipe ImageFrame Packet.

  Raises:
    ValueError:
      i) When "data" is a numpy ndarray, "image_format" is not provided or
        the "data" array is not c_contiguous in the reference mode.
      ii) When "data" is an ImageFrame object, the "image_format" arg doesn't
        match the image format of the "data" ImageFrame object or "copy" is
        explicitly set to False.
    TypeError: If "image format" doesn't match "data" array's data type.

  Examples:
    np_array = np.random.randint(255, size=(321, 123, 3), dtype=np.uint8)
    # Copy mode by default if the data array is writable.
    image_frame_packet = mp.packet_creator.create_image_frame(
        image_format=mp.ImageFormat.SRGB, data=np_array)

    # Make the array unwriteable to trigger the reference mode.
    np_array.flags.writeable = False
    image_frame_packet = mp.packet_creator.create_image_frame(
        image_format=mp.ImageFormat.SRGB, data=np_array)

    image_frame = mp.ImageFrame(image_format=mp.ImageFormat.SRGB, data=np_array)
    image_frame_packet = mp.packet_creator.create_image_frame(image_frame)

  """
  if isinstance(data, image_frame.ImageFrame):
    if image_format is not None and data.image_format != image_format:
      raise ValueError(
          'The provided image_format doesn\'t match the one from the data arg.')
    if copy is not None and not copy:
      raise ValueError(
          'Creating image frame packet by taking a reference of another image frame object is not supported yet.'
      )
    # pylint:disable=protected-access
    return _packet_creator._create_image_frame_from_image_frame(data)
    # pylint:enable=protected-access
  else:
    if image_format is None:
      raise ValueError('Please provide \'image_format\' with \'data\'.')
    # If copy arg is not set, copying the data if it's immutable. Otherwise,
    # take a reference of the immutable data to avoid data copy.
    if copy is None:
      copy = True if data.flags.writeable else False
    if not copy:
      # TODO: Investigate why the first 2 bytes of the data has data
      # corruption when "data" is not c_contiguous.
      if not data.flags.c_contiguous:
        raise ValueError(
            'Reference mode is unavailable if \'data\' is not c_contiguous.')
      if data.flags.writeable:
        warnings.warn(
            '\'data\' is still writeable. Taking a reference of the data to create ImageFrame packet is dangerous.',
            RuntimeWarning, 2)
    # pylint:disable=protected-access
    return _packet_creator._create_image_frame_from_pixel_data(
        image_format, data, copy)
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
