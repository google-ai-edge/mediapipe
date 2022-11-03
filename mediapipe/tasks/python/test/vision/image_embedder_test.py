# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for image embedder."""

import enum
import os
from unittest import mock

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.proto import embedder_options as embedder_options_module
from mediapipe.tasks.python.components.containers import embeddings as embeddings_module
from mediapipe.tasks.python.components.containers import rect
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import image_embedder
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_Rect = rect.Rect
_BaseOptions = base_options_module.BaseOptions
_EmbedderOptions = embedder_options_module.EmbedderOptions
_FloatEmbedding = embeddings_module.FloatEmbedding
_QuantizedEmbedding = embeddings_module.QuantizedEmbedding
_EmbeddingEntry = embeddings_module.EmbeddingEntry
_Embeddings = embeddings_module.Embeddings
_EmbeddingResult = embeddings_module.EmbeddingResult
_Image = image_module.Image
_ImageEmbedder = image_embedder.ImageEmbedder
_ImageEmbedderOptions = image_embedder.ImageEmbedderOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_MODEL_FILE = 'mobilenet_v3_small_100_224_embedder.tflite'
_BURGER_IMAGE_FILE = 'burger.jpg'
_BURGER_CROPPED_IMAGE_FILE = 'burger_crop.jpg'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'
_SIMILARITY_TOLERANCE = 1e-6


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageEmbedderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, _BURGER_IMAGE_FILE)))
    self.test_cropped_image = _Image.create_from_file(
      test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _BURGER_CROPPED_IMAGE_FILE)))
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE))

  def _check_cosine_similarity(self, result0, result1, quantize,
                               expected_similarity):
    # Checks head_index and head_name.
    self.assertEqual(result0.embeddings[0].head_index, 0)
    self.assertEqual(result1.embeddings[0].head_index, 0)
    self.assertEqual(result0.embeddings[0].head_name, 'feature')
    self.assertEqual(result1.embeddings[0].head_name, 'feature')

    # Check embedding sizes.
    def _check_embedding_size(result):
      self.assertLen(result.embeddings, 1)
      embedding_entry = result.embeddings[0].entries[0]
      self.assertLen(embedding_entry.embedding, 1024)
      if quantize:
        self.assertEqual(embedding_entry.embedding.dtype, np.uint8)
      else:
        self.assertEqual(embedding_entry.embedding.dtype, float)

    # Checks results sizes.
    _check_embedding_size(result0)
    _check_embedding_size(result1)

    # Checks cosine similarity.
    similarity = _ImageEmbedder.cosine_similarity(
        result0.embeddings[0].entries[0], result1.embeddings[0].entries[0])
    self.assertAlmostEqual(similarity, expected_similarity,
                           delta=_SIMILARITY_TOLERANCE)

  @parameterized.parameters(
      (False, False, False, ModelFileType.FILE_NAME, 0.925519, -0.2101883),
      (True, False, False, ModelFileType.FILE_NAME, 0.925519, -0.0142344),
      # (False, True, False, ModelFileType.FILE_NAME, 0.926791, 229),
      (False, False, True, ModelFileType.FILE_CONTENT, 0.999931, -0.195062)
  )
  def test_embed(self, l2_normalize, quantize, with_roi, model_file_type,
                 expected_similarity, expected_first_value):
    # Creates embedder.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    embedder_options = _EmbedderOptions(l2_normalize=l2_normalize,
                                        quantize=quantize)
    options = _ImageEmbedderOptions(
        base_options=base_options, embedder_options=embedder_options)
    embedder = _ImageEmbedder.create_from_options(options)

    image_processing_options = None
    if with_roi:
      # Region-of-interest in "burger.jpg" corresponding to "burger_crop.jpg".
      roi = _Rect(left=0, top=0, right=0.833333, bottom=1)
      image_processing_options = _ImageProcessingOptions(roi)

    # Extracts both embeddings.
    image_result = embedder.embed(self.test_image, image_processing_options)
    crop_result = embedder.embed(self.test_cropped_image)

    # Check embedding value.
    self.assertAlmostEqual(image_result.embeddings[0].entries[0].embedding[0],
                           expected_first_value)

    # Checks cosine similarity.
    self._check_cosine_similarity(image_result, crop_result, quantize,
                                  expected_similarity)
    # Closes the embedder explicitly when the classifier is not used in
    # a context.
    embedder.close()

  @parameterized.parameters(
    (False, False, ModelFileType.FILE_NAME, 0.925519),
    (False, False, ModelFileType.FILE_CONTENT, 0.925519))
  def test_embed_in_context(self, l2_normalize, quantize, model_file_type,
                            expected_similarity):
    # Creates embedder.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    embedder_options = _EmbedderOptions(l2_normalize=l2_normalize,
                                        quantize=quantize)
    options = _ImageEmbedderOptions(
      base_options=base_options, embedder_options=embedder_options)

    with _ImageEmbedder.create_from_options(options) as embedder:
      # Extracts both embeddings.
      image_result = embedder.embed(self.test_image)
      crop_result = embedder.embed(self.test_cropped_image)

      # Checks cosine similarity.
      self._check_cosine_similarity(image_result, crop_result, quantize,
                                    expected_similarity)

  def test_missing_result_callback(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM)
    with self.assertRaisesRegex(ValueError,
                                r'result callback must be provided'):
      with _ImageEmbedder.create_from_options(options) as unused_embedder:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=running_mode,
      result_callback=mock.MagicMock())
    with self.assertRaisesRegex(ValueError,
                                r'result callback should not be provided'):
      with _ImageEmbedder.create_from_options(options) as unused_embedder:
        pass

  def test_calling_embed_for_video_in_image_mode(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.IMAGE)
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the video mode'):
        embedder.embed_for_video(self.test_image, 0)

  def test_calling_embed_async_in_image_mode(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.IMAGE)
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the live stream mode'):
        embedder.embed_async(self.test_image, 0)

  def test_calling_embed_in_video_mode(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the image mode'):
        embedder.embed(self.test_image)

  def test_calling_embed_async_in_video_mode(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the live stream mode'):
        embedder.embed_async(self.test_image, 0)

  def test_embed_for_video_with_out_of_order_timestamp(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder:
      unused_result = embedder.embed_for_video(self.test_image, 1)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        embedder.embed_for_video(self.test_image, 0)

  def test_embed_for_video(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder0, \
         _ImageEmbedder.create_from_options(options) as embedder1:
      for timestamp in range(0, 300, 30):
        # Extracts both embeddings.
        image_result = embedder0.embed_for_video(self.test_image, timestamp)
        crop_result = embedder1.embed_for_video(self.test_cropped_image,
                                                timestamp)
        # Checks cosine similarity.
        self._check_cosine_similarity(
            image_result, crop_result, quantize=False,
            expected_similarity=0.925519)

  def test_embed_for_video_succeeds_with_region_of_interest(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder0, \
         _ImageEmbedder.create_from_options(options) as embedder1:
      # Region-of-interest in "burger.jpg" corresponding to "burger_crop.jpg".
      roi = _Rect(left=0, top=0, right=0.833333, bottom=1)
      image_processing_options = _ImageProcessingOptions(roi)

      for timestamp in range(0, 300, 30):
        # Extracts both embeddings.
        image_result = embedder0.embed_for_video(self.test_image, timestamp,
                                                 image_processing_options)
        crop_result = embedder1.embed_for_video(self.test_cropped_image,
                                                timestamp)

        # Checks cosine similarity.
        self._check_cosine_similarity(
            image_result, crop_result, quantize=False,
            expected_similarity=0.999931)

  def test_calling_embed_in_live_stream_mode(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=mock.MagicMock())
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the image mode'):
        embedder.embed(self.test_image)

  def test_calling_classify_for_video_in_live_stream_mode(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=mock.MagicMock())
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the video mode'):
        embedder.embed_for_video(self.test_image, 0)

  def test_classify_async_calls_with_illegal_timestamp(self):
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=mock.MagicMock())
    with _ImageEmbedder.create_from_options(options) as embedder:
      embedder.embed_async(self.test_image, 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        embedder.embed_async(self.test_image, 0)

  def test_embed_async_calls(self):
    # Get the embedding result for the cropped image.
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.IMAGE)
    with _ImageEmbedder.create_from_options(options) as embedder:
      crop_result = embedder.embed(self.test_cropped_image)

    observed_timestamp_ms = -1

    def check_result(result: _EmbeddingResult, output_image: _Image,
                     timestamp_ms: int):
      # Checks cosine similarity.
      self._check_cosine_similarity(result, crop_result, quantize=False,
        expected_similarity=0.925519)
      self.assertTrue(
        np.array_equal(output_image.numpy_view(),
                       self.test_image.numpy_view()))
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=check_result)
    with _ImageEmbedder.create_from_options(options) as embedder:
      for timestamp in range(0, 300, 30):
        embedder.embed_async(self.test_image, timestamp)

  def test_classify_async_succeeds_with_region_of_interest(self):
    # Get the embedding result for the cropped image.
    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.IMAGE)
    with _ImageEmbedder.create_from_options(options) as embedder:
      crop_result = embedder.embed(self.test_cropped_image)

    # Region-of-interest in "burger.jpg" corresponding to "burger_crop.jpg".
    roi = _Rect(left=0, top=0, right=0.833333, bottom=1)
    image_processing_options = _ImageProcessingOptions(roi)
    observed_timestamp_ms = -1

    def check_result(result: _EmbeddingResult, output_image: _Image,
                     timestamp_ms: int):
      # Checks cosine similarity.
      self._check_cosine_similarity(result, crop_result, quantize=False,
                                    expected_similarity=0.999931)
      self.assertTrue(
        np.array_equal(output_image.numpy_view(),
                       self.test_image.numpy_view()))
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    options = _ImageEmbedderOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=check_result)
    with _ImageEmbedder.create_from_options(options) as embedder:
      for timestamp in range(0, 300, 30):
        embedder.embed_async(self.test_image, timestamp,
                             image_processing_options)


if __name__ == '__main__':
  absltest.main()
