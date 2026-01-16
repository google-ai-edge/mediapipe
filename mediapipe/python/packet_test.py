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

"""Tests for mediapipe.python._framework_bindings.packet."""

import gc
import random
import sys
from absl.testing import absltest
import numpy as np

from google.protobuf import text_format
from mediapipe.framework.formats import detection_pb2
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import calculator_graph
from mediapipe.python._framework_bindings import image
from mediapipe.python._framework_bindings import image_frame
from mediapipe.python._framework_bindings import packet

CalculatorGraph = calculator_graph.CalculatorGraph
Image = image.Image
ImageFormat = image_frame.ImageFormat
ImageFrame = image_frame.ImageFrame


class PacketTest(absltest.TestCase):

  def test_empty_packet(self):
    p = packet.Packet()
    self.assertTrue(p.is_empty())

  def test_boolean_packet(self):
    p = packet_creator.create_bool(True)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_bool(p), True)
    self.assertEqual(p.timestamp, 0)

  def test_int_packet(self):
    with self.assertRaisesRegex(OverflowError, 'execeeds the maximum value'):
      p = packet_creator.create_int(2**32)
    p = packet_creator.create_int(42)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_int(p), 42)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_int(np.intc(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_int(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_int8_packet(self):
    with self.assertRaisesRegex(OverflowError, 'execeeds the maximum value'):
      p = packet_creator.create_int8(2**7)
    p = packet_creator.create_int8(2**7 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_int(p), 2**7 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_int8(np.int8(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_int(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_int16_packet(self):
    with self.assertRaisesRegex(OverflowError, 'execeeds the maximum value'):
      p = packet_creator.create_int16(2**15)
    p = packet_creator.create_int16(2**15 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_int(p), 2**15 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_int16(np.int16(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_int(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_int32_packet(self):
    with self.assertRaisesRegex(OverflowError, 'execeeds the maximum value'):
      p = packet_creator.create_int32(2**31)

    p = packet_creator.create_int32(2**31 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_int(p), 2**31 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_int32(np.int32(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_int(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_int64_packet(self):
    p = packet_creator.create_int64(2**63 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_int(p), 2**63 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_int64(np.int64(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_int(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_uint8_packet(self):
    with self.assertRaisesRegex(OverflowError, 'execeeds the maximum value'):
      p = packet_creator.create_uint8(2**8)
    p = packet_creator.create_uint8(2**8 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p), 2**8 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_uint8(np.uint8(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_uint16_packet(self):
    with self.assertRaisesRegex(OverflowError, 'execeeds the maximum value'):
      p = packet_creator.create_uint16(2**16)
    p = packet_creator.create_uint16(2**16 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p), 2**16 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_uint16(np.uint16(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_uint32_packet(self):
    with self.assertRaisesRegex(OverflowError, 'execeeds the maximum value'):
      p = packet_creator.create_uint32(2**32)
    p = packet_creator.create_uint32(2**32 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p), 2**32 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_uint32(np.uint32(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_uint64_packet(self):
    p = packet_creator.create_uint64(2**64 - 1)
    p.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p), 2**64 - 1)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_uint64(np.uint64(1))
    p2.timestamp = 0
    self.assertEqual(packet_getter.get_uint(p2), 1)
    self.assertEqual(p2.timestamp, 0)

  def test_float_packet(self):
    p = packet_creator.create_float(0.42)
    p.timestamp = 0
    self.assertAlmostEqual(packet_getter.get_float(p), 0.42)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_float(float(0.42))
    p2.timestamp = 0
    self.assertAlmostEqual(packet_getter.get_float(p2), 0.42)
    self.assertEqual(p2.timestamp, 0)

  def test_double_packet(self):
    p = packet_creator.create_double(0.42)
    p.timestamp = 0
    self.assertAlmostEqual(packet_getter.get_float(p), 0.42)
    self.assertEqual(p.timestamp, 0)
    p2 = packet_creator.create_double(np.double(0.42))
    p2.timestamp = 0
    self.assertAlmostEqual(packet_getter.get_float(p2), 0.42)
    self.assertEqual(p2.timestamp, 0)

  def test_detection_proto_packet(self):
    detection = detection_pb2.Detection()
    text_format.Parse('score: 0.5', detection)
    p = packet_creator.create_proto(detection).at(100)

  def test_string_packet(self):
    p = packet_creator.create_string('abc').at(100)
    self.assertEqual(packet_getter.get_str(p), 'abc')
    self.assertEqual(p.timestamp, 100)
    p.timestamp = 200
    self.assertEqual(p.timestamp, 200)

  def test_bytes_packet(self):
    p = packet_creator.create_string(b'xd0\xba\xd0').at(300)
    self.assertEqual(packet_getter.get_bytes(p), b'xd0\xba\xd0')
    self.assertEqual(p.timestamp, 300)

  def test_int_array_packet(self):
    p = packet_creator.create_int_array([1, 2, 3]).at(100)
    self.assertEqual(p.timestamp, 100)

  def test_float_array_packet(self):
    p = packet_creator.create_float_array([0.1, 0.2, 0.3]).at(100)
    self.assertEqual(p.timestamp, 100)

  def test_int_vector_packet(self):
    p = packet_creator.create_int_vector([1, 2, 3]).at(100)
    self.assertEqual(packet_getter.get_int_list(p), [1, 2, 3])
    self.assertEqual(p.timestamp, 100)

  def test_float_vector_packet(self):
    p = packet_creator.create_float_vector([0.1, 0.2, 0.3]).at(100)
    output_list = packet_getter.get_float_list(p)
    self.assertAlmostEqual(output_list[0], 0.1)
    self.assertAlmostEqual(output_list[1], 0.2)
    self.assertAlmostEqual(output_list[2], 0.3)
    self.assertEqual(p.timestamp, 100)

  def test_image_vector_packet(self):
    w, h, offset = 80, 40, 10
    mat = np.random.randint(2**8 - 1, size=(h, w, 3), dtype=np.uint8)
    p = packet_creator.create_image_vector([
        Image(image_format=ImageFormat.SRGB, data=mat),
        Image(
            image_format=ImageFormat.SRGB,
            data=np.ascontiguousarray(mat[offset:-offset, offset:-offset, :]))
    ]).at(100)
    output_list = packet_getter.get_image_list(p)
    self.assertLen(output_list, 2)
    self.assertTrue(np.array_equal(output_list[0].numpy_view(), mat))
    self.assertTrue(
        np.array_equal(output_list[1].numpy_view(), mat[offset:-offset,
                                                        offset:-offset, :]))
    self.assertEqual(p.timestamp, 100)

  def test_image_frame_vector_packet(self):
    mat_rgb = np.random.randint(2**8 - 1, size=(30, 40, 3), dtype=np.uint8)
    mat_float = np.random.random(size=(40, 30)).astype(np.float32)
    p = packet_creator.create_image_frame_vector([
        ImageFrame(image_format=ImageFormat.SRGB, data=mat_rgb),
        ImageFrame(image_format=ImageFormat.VEC32F1, data=mat_float),
    ]).at(100)
    output_list = packet_getter.get_image_frame_list(p)
    self.assertLen(output_list, 2)
    self.assertTrue(np.array_equal(output_list[0].numpy_view(), mat_rgb))
    self.assertTrue(np.array_equal(output_list[1].numpy_view(), mat_float))
    self.assertEqual(p.timestamp, 100)

  def test_get_image_frame_list_packet_capture(self):
    h, w, c = 30, 40, 3
    p = packet_creator.create_image_frame_vector([
        ImageFrame(
            image_format=ImageFormat.SRGB,
            data=np.ones((h, w, c), dtype=np.uint8),
        ),
    ]).at(100)
    output_list = packet_getter.get_image_frame_list(p)
    # Even if the packet variable p gets deleted, the packet object still exists
    # because it is captured by the deleter of the ImageFrame in the returned
    # output_list.
    del p
    gc.collect()
    self.assertLen(output_list, 1)
    self.assertEqual(output_list[0].image_format, ImageFormat.SRGB)
    self.assertTrue(
        np.array_equal(
            output_list[0].numpy_view(), np.ones((h, w, c), dtype=np.uint8)
        )
    )

  def test_create_image_frame_vector_type_error(self):
    with self.assertRaisesRegex(TypeError, 'not an ImageFrame'):
      packet_creator.create_image_frame_vector([None])

  def test_string_vector_packet(self):
    p = packet_creator.create_string_vector(['a', 'b', 'c']).at(100)
    output_list = packet_getter.get_str_list(p)
    self.assertEqual(output_list[0], 'a')
    self.assertEqual(output_list[1], 'b')
    self.assertEqual(output_list[2], 'c')
    self.assertEqual(p.timestamp, 100)

  def test_packet_vector_packet(self):
    p = packet_creator.create_packet_vector([
        packet_creator.create_float(0.42),
        packet_creator.create_int(42),
        packet_creator.create_string('42')
    ]).at(100)
    output_list = packet_getter.get_packet_list(p)
    self.assertAlmostEqual(packet_getter.get_float(output_list[0]), 0.42)
    self.assertEqual(packet_getter.get_int(output_list[1]), 42)
    self.assertEqual(packet_getter.get_str(output_list[2]), '42')
    self.assertEqual(p.timestamp, 100)

  def test_string_to_packet_map_packet(self):
    p = packet_creator.create_string_to_packet_map({
        'float': packet_creator.create_float(0.42),
        'int': packet_creator.create_int(42),
        'string': packet_creator.create_string('42')
    }).at(100)
    output_list = packet_getter.get_str_to_packet_dict(p)
    self.assertAlmostEqual(packet_getter.get_float(output_list['float']), 0.42)
    self.assertEqual(packet_getter.get_int(output_list['int']), 42)
    self.assertEqual(packet_getter.get_str(output_list['string']), '42')
    self.assertEqual(p.timestamp, 100)

  def test_uint8_image_packet(self):
    uint8_img = np.random.randint(
        2**8 - 1,
        size=(random.randrange(3, 100), random.randrange(3, 100), 3),
        dtype=np.uint8)
    image_frame_packet = packet_creator.create_image_frame(
        image_frame.ImageFrame(
            image_format=image_frame.ImageFormat.SRGB, data=uint8_img))
    output_image_frame = packet_getter.get_image_frame(image_frame_packet)
    self.assertTrue(np.array_equal(output_image_frame.numpy_view(), uint8_img))
    image_packet = packet_creator.create_image(
        Image(image_format=ImageFormat.SRGB, data=uint8_img))
    output_image = packet_getter.get_image(image_packet)
    self.assertTrue(np.array_equal(output_image.numpy_view(), uint8_img))

  def test_uint16_image_packet(self):
    uint16_img = np.random.randint(
        2**16 - 1,
        size=(random.randrange(3, 100), random.randrange(3, 100), 4),
        dtype=np.uint16)
    image_frame_packet = packet_creator.create_image_frame(
        ImageFrame(image_format=ImageFormat.SRGBA64, data=uint16_img))
    output_image_frame = packet_getter.get_image_frame(image_frame_packet)
    self.assertTrue(np.array_equal(output_image_frame.numpy_view(), uint16_img))
    image_packet = packet_creator.create_image(
        Image(image_format=ImageFormat.SRGBA64, data=uint16_img))
    output_image = packet_getter.get_image(image_packet)
    self.assertTrue(np.array_equal(output_image.numpy_view(), uint16_img))

  def test_float_image_frame_packet(self):
    float_img = np.float32(
        np.random.random_sample(
            (random.randrange(3, 100), random.randrange(3, 100), 2)))
    image_frame_packet = packet_creator.create_image_frame(
        ImageFrame(image_format=ImageFormat.VEC32F2, data=float_img))
    output_image_frame = packet_getter.get_image_frame(image_frame_packet)
    self.assertTrue(np.allclose(output_image_frame.numpy_view(), float_img))
    image_packet = packet_creator.create_image(
        Image(image_format=ImageFormat.VEC32F2, data=float_img))
    output_image = packet_getter.get_image(image_packet)
    self.assertTrue(np.array_equal(output_image.numpy_view(), float_img))

  def test_image_frame_packet_creation_copy_mode(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    rgb_data = np.random.randint(255, size=(h, w, channels), dtype=np.uint8)
    # rgb_data is c_contiguous.
    self.assertTrue(rgb_data.flags.c_contiguous)
    initial_ref_count = sys.getrefcount(rgb_data)
    p = packet_creator.create_image_frame(
        image_format=ImageFormat.SRGB, data=rgb_data)
    # copy mode doesn't increase the ref count of the data.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

    rgb_data = rgb_data[:, :, ::-1]
    # rgb_data is now not c_contiguous. But, copy mode shouldn't be affected.
    self.assertFalse(rgb_data.flags.c_contiguous)
    initial_ref_count = sys.getrefcount(rgb_data)
    p = packet_creator.create_image_frame(
        image_format=ImageFormat.SRGB, data=rgb_data)
    # copy mode doesn't increase the ref count of the data.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

    output_frame = packet_getter.get_image_frame(p)
    self.assertEqual(output_frame.height, h)
    self.assertEqual(output_frame.width, w)
    self.assertEqual(output_frame.channels, channels)
    self.assertTrue(np.array_equal(output_frame.numpy_view(), rgb_data))

    del p
    del output_frame
    gc.collect()
    # Destroying the packet also doesn't affect the ref count becuase of the
    # copy mode.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

  def test_image_frame_packet_creation_reference_mode(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    rgb_data = np.random.randint(255, size=(h, w, channels), dtype=np.uint8)
    rgb_data.flags.writeable = False
    initial_ref_count = sys.getrefcount(rgb_data)
    image_frame_packet = packet_creator.create_image_frame(
        image_format=ImageFormat.SRGB, data=rgb_data)
    # Reference mode increase the ref count of the rgb_data by 1.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count + 1)
    del image_frame_packet
    gc.collect()
    # Deleting image_frame_packet should decrese the ref count of rgb_data by 1.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)
    rgb_data_copy = np.copy(rgb_data)
    # rgb_data_copy is a copy of rgb_data and should not increase the ref count.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)
    text_config = """
      node {
        calculator: 'PassThroughCalculator'
        input_side_packet: "in"
        output_side_packet: "out"
      }
    """
    graph = CalculatorGraph(graph_config=text_config)
    graph.start_run(
        input_side_packets={
            'in':
                packet_creator.create_image_frame(
                    image_format=ImageFormat.SRGB, data=rgb_data)
        })
    # reference mode increase the ref count of the rgb_data by 1.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count + 1)
    graph.wait_until_done()
    output_packet = graph.get_output_side_packet('out')
    del rgb_data
    del graph
    gc.collect()
    # The pixel data of the output image frame packet should still be valid
    # after the graph and the original rgb_data data are deleted.
    self.assertTrue(
        np.array_equal(
            packet_getter.get_image_frame(output_packet).numpy_view(),
            rgb_data_copy))

  def test_image_frame_packet_copy_creation_with_cropping(self):
    w, h, channels = random.randrange(40, 100), random.randrange(40, 100), 3
    channels, offset = 3, 10
    rgb_data = np.random.randint(255, size=(h, w, channels), dtype=np.uint8)
    initial_ref_count = sys.getrefcount(rgb_data)
    p = packet_creator.create_image_frame(
        image_format=ImageFormat.SRGB,
        data=rgb_data[offset:-offset, offset:-offset, :])
    # copy mode doesn't increase the ref count of the data.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)
    output_frame = packet_getter.get_image_frame(p)
    self.assertEqual(output_frame.height, h - 2 * offset)
    self.assertEqual(output_frame.width, w - 2 * offset)
    self.assertEqual(output_frame.channels, channels)
    self.assertTrue(
        np.array_equal(rgb_data[offset:-offset, offset:-offset, :],
                       output_frame.numpy_view()))
    del p
    del output_frame
    gc.collect()
    # Destroying the packet also doesn't affect the ref count becuase of the
    # copy mode.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

  def test_image_packet_creation_copy_mode(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    rgb_data = np.random.randint(255, size=(h, w, channels), dtype=np.uint8)
    # rgb_data is c_contiguous.
    self.assertTrue(rgb_data.flags.c_contiguous)
    initial_ref_count = sys.getrefcount(rgb_data)
    p = packet_creator.create_image(
        image_format=ImageFormat.SRGB, data=rgb_data)
    # copy mode doesn't increase the ref count of the data.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

    rgb_data = rgb_data[:, :, ::-1]
    # rgb_data is now not c_contiguous. But, copy mode shouldn't be affected.
    self.assertFalse(rgb_data.flags.c_contiguous)
    initial_ref_count = sys.getrefcount(rgb_data)
    p = packet_creator.create_image(
        image_format=ImageFormat.SRGB, data=rgb_data)
    # copy mode doesn't increase the ref count of the data.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

    output_image = packet_getter.get_image(p)
    self.assertEqual(output_image.height, h)
    self.assertEqual(output_image.width, w)
    self.assertEqual(output_image.channels, channels)
    self.assertTrue(np.array_equal(output_image.numpy_view(), rgb_data))

    del p
    del output_image
    gc.collect()
    # Destroying the packet also doesn't affect the ref count becuase of the
    # copy mode.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

  def test_image_packet_creation_reference_mode(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    rgb_data = np.random.randint(255, size=(h, w, channels), dtype=np.uint8)
    rgb_data.flags.writeable = False
    initial_ref_count = sys.getrefcount(rgb_data)
    image_packet = packet_creator.create_image(
        image_format=ImageFormat.SRGB, data=rgb_data)
    # Reference mode increase the ref count of the rgb_data by 1.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count + 1)
    del image_packet
    gc.collect()
    # Deleting image_packet should decrese the ref count of rgb_data by 1.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)
    rgb_data_copy = np.copy(rgb_data)
    # rgb_data_copy is a copy of rgb_data and should not increase the ref count.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)
    text_config = """
      node {
        calculator: 'PassThroughCalculator'
        input_side_packet: "in"
        output_side_packet: "out"
      }
    """
    graph = CalculatorGraph(graph_config=text_config)
    graph.start_run(
        input_side_packets={
            'in':
                packet_creator.create_image(
                    image_format=ImageFormat.SRGB, data=rgb_data)
        })
    # reference mode increase the ref count of the rgb_data by 1.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count + 1)
    graph.wait_until_done()
    output_packet = graph.get_output_side_packet('out')
    del rgb_data
    del graph
    gc.collect()
    # The pixel data of the output image frame packet should still be valid
    # after the graph and the original rgb_data data are deleted.
    self.assertTrue(
        np.array_equal(
            packet_getter.get_image(output_packet).numpy_view(), rgb_data_copy))

  def test_image_packet_copy_creation_with_cropping(self):
    w, h, channels = random.randrange(40, 100), random.randrange(40, 100), 3
    channels, offset = 3, 10
    rgb_data = np.random.randint(255, size=(h, w, channels), dtype=np.uint8)
    initial_ref_count = sys.getrefcount(rgb_data)
    p = packet_creator.create_image(
        image_format=ImageFormat.SRGB,
        data=rgb_data[offset:-offset, offset:-offset, :])
    # copy mode doesn't increase the ref count of the data.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)
    output_image = packet_getter.get_image(p)
    self.assertEqual(output_image.height, h - 2 * offset)
    self.assertEqual(output_image.width, w - 2 * offset)
    self.assertEqual(output_image.channels, channels)
    self.assertTrue(
        np.array_equal(rgb_data[offset:-offset, offset:-offset, :],
                       output_image.numpy_view()))
    del p
    del output_image
    gc.collect()
    # Destroying the packet also doesn't affect the ref count becuase of the
    # copy mode.
    self.assertEqual(sys.getrefcount(rgb_data), initial_ref_count)

  def test_matrix_packet(self):
    np_matrix = np.array([[.1, .2, .3], [.4, .5, .6]])
    initial_ref_count = sys.getrefcount(np_matrix)
    p = packet_creator.create_matrix(np_matrix)
    # Copy mode should not increase the ref count of np_matrix.
    self.assertEqual(initial_ref_count, sys.getrefcount(np_matrix))
    output_matrix = packet_getter.get_matrix(p)
    del np_matrix
    gc.collect()
    self.assertTrue(
        np.allclose(output_matrix, np.array([[.1, .2, .3], [.4, .5, .6]])))

  def test_matrix_packet_with_non_c_contiguous_data(self):
    np_matrix = np.array([[.1, .2, .3], [.4, .5, .6]])[:, ::-1]
    # np_matrix is not c_contiguous.
    self.assertFalse(np_matrix.flags.c_contiguous)
    p = packet_creator.create_matrix(np_matrix)
    initial_ref_count = sys.getrefcount(np_matrix)
    # Copy mode should not increase the ref count of np_matrix.
    self.assertEqual(initial_ref_count, sys.getrefcount(np_matrix))
    output_matrix = packet_getter.get_matrix(p)
    del np_matrix
    gc.collect()
    self.assertTrue(
        np.allclose(output_matrix,
                    np.array([[.1, .2, .3], [.4, .5, .6]])[:, ::-1]))

if __name__ == '__main__':
  absltest.main()
