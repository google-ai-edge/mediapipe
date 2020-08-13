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
"""The public facing packet getter APIs."""

from typing import List, Type

from google.protobuf import message
from google.protobuf import symbol_database
from mediapipe.python._framework_bindings import _packet_getter
from mediapipe.python._framework_bindings import packet as mp_packet

get_str = _packet_getter.get_str
get_bytes = _packet_getter.get_bytes
get_bool = _packet_getter.get_bool
get_int = _packet_getter.get_int
get_uint = _packet_getter.get_uint
get_float = _packet_getter.get_float
get_int_list = _packet_getter.get_int_list
get_float_list = _packet_getter.get_float_list
get_str_list = _packet_getter.get_str_list
get_packet_list = _packet_getter.get_packet_list
get_str_to_packet_dict = _packet_getter.get_str_to_packet_dict
get_image_frame = _packet_getter.get_image_frame
get_matrix = _packet_getter.get_matrix


def get_proto(packet: mp_packet.Packet) -> Type[message.Message]:
  """Get the content of a MediaPipe proto Packet as a proto message.

  Args:
    packet: A MediaPipe proto Packet.

  Returns:
    A proto message.

  Raises:
    TypeError: If the message descriptor can't be found by type name.

  Examples:
    detection = detection_pb2.Detection()
    text_format.Parse('score: 0.5', detection)
    proto_packet = mp.packet_creator.create_proto(detection)
    output_proto = mp.packet_getter.get_proto(proto_packet)
  """
  # pylint:disable=protected-access
  proto_type_name = _packet_getter._get_proto_type_name(packet)
  # pylint:enable=protected-access
  try:
    descriptor = symbol_database.Default().pool.FindMessageTypeByName(
        proto_type_name)
  except KeyError:
    raise TypeError('Can not find message descriptor by type name: %s' %
                    proto_type_name)

  message_class = symbol_database.Default().GetPrototype(descriptor)
  # pylint:disable=protected-access
  serialized_proto = _packet_getter._get_serialized_proto(packet)
  # pylint:enable=protected-access
  proto_message = message_class()
  proto_message.ParseFromString(serialized_proto)
  return proto_message


def get_proto_list(packet: mp_packet.Packet) -> List[message.Message]:
  """Get the content of a MediaPipe proto vector Packet as a proto message list.

  Args:
    packet: A MediaPipe proto vector Packet.

  Returns:
    A proto message list.

  Raises:
    TypeError: If the message descriptor can't be found by type name.

  Examples:
    proto_list = mp.packet_getter.get_proto_list(protos_packet)
  """
  # pylint:disable=protected-access
  vector_size = _packet_getter._get_proto_vector_size(packet)
  # pylint:enable=protected-access
  # Return empty list if the proto vector is empty.
  if vector_size == 0:
    return []

  # pylint:disable=protected-access
  proto_type_name = _packet_getter._get_proto_vector_element_type_name(packet)
  # pylint:enable=protected-access
  try:
    descriptor = symbol_database.Default().pool.FindMessageTypeByName(
        proto_type_name)
  except KeyError:
    raise TypeError('Can not find message descriptor by type name: %s' %
                    proto_type_name)
  message_class = symbol_database.Default().GetPrototype(descriptor)
  # pylint:disable=protected-access
  serialized_protos = _packet_getter._get_serialized_proto_list(packet)
  # pylint:enable=protected-access
  proto_message_list = []
  for serialized_proto in serialized_protos:
    proto_message = message_class()
    proto_message.ParseFromString(serialized_proto)
    proto_message_list.append(proto_message)
  return proto_message_list
