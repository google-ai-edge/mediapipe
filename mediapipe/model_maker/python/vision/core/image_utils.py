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
# ==============================================================================
"""Utilities for Images."""

import tensorflow as tf


def load_image(path: str) -> tf.Tensor:
  """Loads a jpeg/png image and returns an image tensor."""
  image_raw = tf.io.read_file(path)
  image_tensor = tf.cond(
      tf.io.is_jpeg(image_raw),
      lambda: tf.io.decode_jpeg(image_raw, channels=3),
      lambda: tf.io.decode_png(image_raw, channels=3),
  )
  return image_tensor
