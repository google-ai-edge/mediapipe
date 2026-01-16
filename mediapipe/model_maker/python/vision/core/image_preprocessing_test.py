# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.vision.core import image_preprocessing


def _get_preprocessed_image(preprocessor, is_training=False):
  image_placeholder = tf.compat.v1.placeholder(tf.uint8, [24, 24, 3])
  label_placeholder = tf.compat.v1.placeholder(tf.int32, [1])
  image_tensor, _ = preprocessor(image_placeholder, label_placeholder,
                                 is_training)

  with tf.compat.v1.Session() as sess:
    input_image = np.arange(24 * 24 * 3, dtype=np.uint8).reshape([24, 24, 3])
    image = sess.run(
        image_tensor,
        feed_dict={
            image_placeholder: input_image,
            label_placeholder: [0]
        })
    return image


class PreprocessorTest(tf.test.TestCase):

  def test_preprocess_without_augmentation(self):
    preprocessor = image_preprocessing.Preprocessor(input_shape=[2, 2],
                                                    num_classes=2,
                                                    mean_rgb=[0.0],
                                                    stddev_rgb=[255.0],
                                                    use_augmentation=False)
    actual_image = np.array([[[0., 0.00392157, 0.00784314],
                              [0.14117648, 0.14509805, 0.14901961]],
                             [[0.37647063, 0.3803922, 0.38431376],
                              [0.5176471, 0.52156866, 0.5254902]]])

    image = _get_preprocessed_image(preprocessor)
    self.assertTrue(np.allclose(image, actual_image, atol=1e-05))

  def test_preprocess_with_augmentation(self):
    image_preprocessing.CROP_PADDING = 1
    preprocessor = image_preprocessing.Preprocessor(input_shape=[2, 2],
                                                    num_classes=2,
                                                    mean_rgb=[0.0],
                                                    stddev_rgb=[255.0],
                                                    use_augmentation=True)
    # Tests validation image.
    actual_eval_image = np.array([[[0.17254902, 0.1764706, 0.18039216],
                                   [0.26666668, 0.27058825, 0.27450982]],
                                  [[0.42352945, 0.427451, 0.43137258],
                                   [0.5176471, 0.52156866, 0.5254902]]])

    image = _get_preprocessed_image(preprocessor, is_training=False)
    self.assertTrue(np.allclose(image, actual_eval_image, atol=1e-05))

    # Tests training image.
    image1 = _get_preprocessed_image(preprocessor, is_training=True)
    image2 = _get_preprocessed_image(preprocessor, is_training=True)
    self.assertFalse(np.allclose(image1, image2, atol=1e-05))
    self.assertEqual(image1.shape, (2, 2, 3))
    self.assertEqual(image2.shape, (2, 2, 3))


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
