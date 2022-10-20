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
from unittest import mock

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.proto import embedder_options as embedder_options_module
from mediapipe.tasks.python.components.containers import embeddings as embeddings_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import image_embedder
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_NormalizedRect = rect_module.NormalizedRect
_BaseOptions = base_options_module.BaseOptions
_EmbedderOptions = embedder_options_module.EmbedderOptions
_FloatEmbedding = embeddings_module.FloatEmbedding
_QuantizedEmbedding = embeddings_module.QuantizedEmbedding
_ClassificationEntry = embeddings_module.EmbeddingEntry
_Classifications = embeddings_module.Embeddings
_ClassificationResult = embeddings_module.EmbeddingResult
_Image = image_module.Image
_ImageEmbedder = image_embedder.ImageEmbedder
_ImageEmbedderOptions = image_embedder.ImageEmbedderOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_MODEL_FILE = 'mobilenet_v3_small_100_224_embedder.tflite'
_IMAGE_FILE = 'burger.jpg'
_ALLOW_LIST = ['cheeseburger', 'guacamole']
_DENY_LIST = ['cheeseburger']
_SCORE_THRESHOLD = 0.5
_MAX_RESULTS = 3


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageClassifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_IMAGE_FILE))
    self.model_path = test_utils.get_test_data_path(_MODEL_FILE)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, False, False),
      (ModelFileType.FILE_CONTENT, False, False))
  def test_embed(self, model_file_type, l2_normalize, quantize):
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

    # Performs image embedding extraction on the input.
    image_result = embedder.embed(self.test_image)

    # TODO: Verify results.

    # Closes the embedder explicitly when the classifier is not used in
    # a context.
    embedder.close()


if __name__ == '__main__':
  absltest.main()
