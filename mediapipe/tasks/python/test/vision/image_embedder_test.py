# Copyright 2022 The MediaPipe Authors.
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
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import image_embedder
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_RectF = rect_module.RectF
_BaseOptions = base_options_module.BaseOptions
_Embedding = embedding_result_module.Embedding
_Image = image_module.Image
_ImageEmbedder = image_embedder.ImageEmbedder
_ImageEmbedderOptions = image_embedder.ImageEmbedderOptions
_ImageEmbedderResult = image_embedder.ImageEmbedderResult
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_MODEL_FILE = 'mobilenet_v3_small_100_224_embedder.tflite'
_BURGER_IMAGE_FILE = 'burger.jpg'
_BURGER_CROPPED_IMAGE_FILE = 'burger_crop.jpg'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'
# Tolerance for embedding vector coordinate values.
_EPSILON = 1e-4
# Tolerance for cosine similarity evaluation.
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

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _ImageEmbedder.create_from_model_path(self.model_path) as embedder:
      self.assertIsInstance(embedder, _ImageEmbedder)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ImageEmbedderOptions(base_options=base_options)
    with _ImageEmbedder.create_from_options(options) as embedder:
      self.assertIsInstance(embedder, _ImageEmbedder)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        FileNotFoundError,
        'Unable to open file at /path/to/invalid/model.tflite',
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite')
      options = _ImageEmbedderOptions(base_options=base_options)
      embedder = _ImageEmbedder.create_from_options(options)
      embedder.close()

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _ImageEmbedderOptions(base_options=base_options)
      embedder = _ImageEmbedder.create_from_options(options)
      self.assertIsInstance(embedder, _ImageEmbedder)
      embedder.close()

  def _check_embedding_value(self, result, expected_first_value):
    # Check embedding first value.
    self.assertAlmostEqual(
        result.embeddings[0].embedding[0], expected_first_value, delta=_EPSILON)

  def _check_embedding_size(self, result, quantize, expected_embedding_size):
    # Check embedding size.
    self.assertLen(result.embeddings, 1)
    embedding_result = result.embeddings[0]
    self.assertLen(embedding_result.embedding, expected_embedding_size)
    if quantize:
      self.assertEqual(embedding_result.embedding.dtype, np.uint8)
    else:
      self.assertEqual(embedding_result.embedding.dtype, float)

  def _check_cosine_similarity(self, result0, result1, expected_similarity):
    # Checks cosine similarity.
    similarity = _ImageEmbedder.cosine_similarity(result0.embeddings[0],
                                                  result1.embeddings[0])
    self.assertAlmostEqual(
        similarity, expected_similarity, delta=_SIMILARITY_TOLERANCE)

  @parameterized.parameters(
      (
          False,
          False,
          False,
          ModelFileType.FILE_NAME,
          0.925519,
          1024,
          (-0.2101883, -0.193027),
      ),
      (
          True,
          False,
          False,
          ModelFileType.FILE_NAME,
          0.925519,
          1024,
          (-0.0142344, -0.0131606),
      ),
      (
          False,
          True,
          False,
          ModelFileType.FILE_NAME,
          0.926791,
          1024,
          (229, 231),
      ),
      (
          False,
          False,
          True,
          ModelFileType.FILE_CONTENT,
          0.999931,
          1024,
          (-0.195062, -0.193027),
      ),
  )
  def test_embed(self, l2_normalize, quantize, with_roi, model_file_type,
                 expected_similarity, expected_size, expected_first_values):
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

    options = _ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)
    embedder = _ImageEmbedder.create_from_options(options)

    image_processing_options = None
    if with_roi:
      # Region-of-interest in "burger.jpg" corresponding to "burger_crop.jpg".
      roi = _RectF(left=0.0, top=0.0, right=0.833333, bottom=1.0)
      image_processing_options = _ImageProcessingOptions(roi)

    # Extracts both embeddings.
    image_result = embedder.embed(self.test_image, image_processing_options)
    crop_result = embedder.embed(self.test_cropped_image)

    # Checks embeddings and cosine similarity.
    expected_result0_value, expected_result1_value = expected_first_values
    self._check_embedding_size(image_result, quantize, expected_size)
    self._check_embedding_size(crop_result, quantize, expected_size)
    self._check_embedding_value(image_result, expected_result0_value)
    self._check_embedding_value(crop_result, expected_result1_value)
    self._check_cosine_similarity(image_result, crop_result,
                                  expected_similarity)
    # Closes the embedder explicitly when the embedder is not used in
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

    options = _ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)

    with _ImageEmbedder.create_from_options(options) as embedder:
      # Extracts both embeddings.
      image_result = embedder.embed(self.test_image)
      crop_result = embedder.embed(self.test_cropped_image)

      # Checks cosine similarity.
      self._check_cosine_similarity(image_result, crop_result,
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
      with self.assertRaises(ValueError):
        embedder.embed_for_video(self.test_image, 0)

  def test_calling_embed_async_in_image_mode(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE)
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaises(ValueError):
        embedder.embed_async(self.test_image, 0)

  def test_calling_embed_in_video_mode(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaises(ValueError):
        embedder.embed(self.test_image)

  def test_calling_embed_async_in_video_mode(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaises(ValueError):
        embedder.embed_async(self.test_image, 0)

  def test_embed_for_video_with_out_of_order_timestamp(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder:
      unused_result = embedder.embed_for_video(self.test_image, 1)
      with self.assertRaises(ValueError):
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
            image_result, crop_result, expected_similarity=0.925519)

  def test_embed_for_video_succeeds_with_region_of_interest(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO)
    with _ImageEmbedder.create_from_options(options) as embedder0, \
         _ImageEmbedder.create_from_options(options) as embedder1:
      # Region-of-interest in "burger.jpg" corresponding to "burger_crop.jpg".
      roi = _RectF(left=0.0, top=0.0, right=0.833333, bottom=1.0)
      image_processing_options = _ImageProcessingOptions(roi)

      for timestamp in range(0, 300, 30):
        # Extracts both embeddings.
        image_result = embedder0.embed_for_video(self.test_image, timestamp,
                                                 image_processing_options)
        crop_result = embedder1.embed_for_video(self.test_cropped_image,
                                                timestamp)

        # Checks cosine similarity.
        self._check_cosine_similarity(
            image_result, crop_result, expected_similarity=0.999931)

  def test_calling_embed_in_live_stream_mode(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock())
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaises(ValueError):
        embedder.embed(self.test_image)

  def test_calling_embed_for_video_in_live_stream_mode(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock())
    with _ImageEmbedder.create_from_options(options) as embedder:
      with self.assertRaises(ValueError):
        embedder.embed_for_video(self.test_image, 0)

  def test_embed_async_calls_with_illegal_timestamp(self):
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock())
    with _ImageEmbedder.create_from_options(options) as embedder:
      embedder.embed_async(self.test_image, 100)
      with self.assertRaises(ValueError):
        embedder.embed_async(self.test_image, 0)

  def test_embed_async_calls(self):
    # Get the embedding result for the cropped image.
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE)
    with _ImageEmbedder.create_from_options(options) as embedder:
      crop_result = embedder.embed(self.test_cropped_image)

    callback_event = threading.Event()
    callback_exception: None | Exception = None
    observed_timestamp_ms = -1

    def check_result(result: _ImageEmbedderResult, output_image: _Image,
                     timestamp_ms: int):
      nonlocal callback_exception, observed_timestamp_ms
      try:
        # Checks cosine similarity.
        self._check_cosine_similarity(
            result, crop_result, expected_similarity=0.925519
        )
        self.assertTrue(
            np.array_equal(
                output_image.numpy_view(), self.test_image.numpy_view()
            )
        )
        self.assertLess(observed_timestamp_ms, timestamp_ms)
        observed_timestamp_ms = timestamp_ms
      except AssertionError as e:
        callback_exception = e
      finally:
        callback_event.set()

    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=check_result)
    with _ImageEmbedder.create_from_options(options) as embedder:
      for timestamp in range(0, 300, 30):
        embedder.embed_async(self.test_image, timestamp)
        callback_event.wait()
        if callback_exception is not None:
          raise callback_exception
        callback_event.clear()

  def test_embed_async_succeeds_with_region_of_interest(self):
    # Get the embedding result for the cropped image.
    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE)
    with _ImageEmbedder.create_from_options(options) as embedder:
      crop_result = embedder.embed(self.test_cropped_image)

    # Region-of-interest in "burger.jpg" corresponding to "burger_crop.jpg".
    roi = _RectF(left=0.0, top=0.0, right=0.833333, bottom=1.0)
    image_processing_options = _ImageProcessingOptions(roi)
    callback_event = threading.Event()
    callback_exception: None | Exception = None
    observed_timestamp_ms = -1

    def check_result(result: _ImageEmbedderResult, output_image: _Image,
                     timestamp_ms: int):
      nonlocal callback_exception, observed_timestamp_ms
      try:
        # Checks cosine similarity.
        self._check_cosine_similarity(
            result, crop_result, expected_similarity=0.999931
        )
        self.assertTrue(
            np.array_equal(
                output_image.numpy_view(), self.test_image.numpy_view()
            )
        )
        self.assertLess(observed_timestamp_ms, timestamp_ms)
        observed_timestamp_ms = timestamp_ms
      except AssertionError as e:
        callback_exception = e
      finally:
        callback_event.set()

    options = _ImageEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=check_result)
    with _ImageEmbedder.create_from_options(options) as embedder:
      for timestamp in range(0, 300, 30):
        embedder.embed_async(self.test_image, timestamp,
                             image_processing_options)
        callback_event.wait()
        if callback_exception is not None:
          raise callback_exception
        callback_event.clear()


if __name__ == '__main__':
  absltest.main()
