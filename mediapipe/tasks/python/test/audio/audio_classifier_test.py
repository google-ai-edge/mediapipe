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
"""Tests for audio classifier."""

import os
from typing import List, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy.io import wavfile

from mediapipe.tasks.python.audio import audio_classifier
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.audio.core import audio_task_running_mode
from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils

_AudioClassifier = audio_classifier.AudioClassifier
_AudioClassifierOptions = audio_classifier.AudioClassifierOptions
_AudioClassifierResult = classification_result_module.ClassificationResult
_AudioData = audio_data_module.AudioData
_AudioRecord = audio_record.AudioRecord
_BaseOptions = base_options_module.BaseOptions
_RUNNING_MODE = audio_task_running_mode.AudioTaskRunningMode

_YAMNET_MODEL_FILE = 'yamnet_audio_classifier_with_metadata.tflite'
_YAMNET_MODEL_SAMPLE_RATE = 16000
_TWO_HEADS_MODEL_FILE = 'two_heads.tflite'
_SPEECH_WAV_16K_MONO = 'speech_16000_hz_mono.wav'
_SPEECH_WAV_48K_MONO = 'speech_48000_hz_mono.wav'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/audio'
_TWO_HEADS_WAV_16K_MONO = 'two_heads_16000_hz_mono.wav'
_TWO_HEADS_WAV_44K_MONO = 'two_heads_44100_hz_mono.wav'
_YAMNET_NUM_OF_SAMPLES = 15600
_MILLISECONDS_PER_SECOND = 1000


class AudioClassifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.yamnet_model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _YAMNET_MODEL_FILE))
    self.two_heads_model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _TWO_HEADS_MODEL_FILE))

  def _read_wav_file(self, file_name) -> _AudioData:
    sample_rate, buffer = wavfile.read(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, file_name)))
    return _AudioData.create_from_array(
        buffer.astype(float) / np.iinfo(np.int16).max, sample_rate)

  def _read_wav_file_as_stream(self, file_name) -> List[Tuple[_AudioData, int]]:
    sample_rate, buffer = wavfile.read(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, file_name)))
    audio_data_list = []
    start = 0
    step_size = _YAMNET_NUM_OF_SAMPLES * sample_rate / _YAMNET_MODEL_SAMPLE_RATE
    while start < len(buffer):
      end = min(start + (int)(step_size), len(buffer))
      audio_data_list.append((_AudioData.create_from_array(
          buffer[start:end].astype(float) / np.iinfo(np.int16).max,
          sample_rate), (int)(start / sample_rate * _MILLISECONDS_PER_SECOND)))
      start = end
    return audio_data_list

  # TODO: Compares the exact score values to capture unexpected
  # changes in the inference pipeline.
  def _check_yamnet_result(
      self,
      classification_result_list: List[_AudioClassifierResult],
      expected_num_categories=521):
    self.assertLen(classification_result_list, 5)
    for idx, timestamp in enumerate([0, 975, 1950, 2925]):
      classification_result = classification_result_list[idx]
      self.assertEqual(classification_result.timestamp_ms, timestamp)
      self.assertLen(classification_result.classifications, 1)
      classifcation = classification_result.classifications[0]
      self.assertEqual(classifcation.head_index, 0)
      self.assertEqual(classifcation.head_name, 'scores')
      self.assertLen(classifcation.categories, expected_num_categories)
      audio_category = classifcation.categories[0]
      self.assertEqual(audio_category.index, 0)
      self.assertEqual(audio_category.category_name, 'Speech')
      self.assertGreater(audio_category.score, 0.9)

  # TODO: Compares the exact score values to capture unexpected
  # changes in the inference pipeline.
  def _check_two_heads_result(
      self,
      classification_result_list: List[_AudioClassifierResult],
      first_head_expected_num_categories=521,
      second_head_expected_num_categories=5):
    self.assertGreaterEqual(len(classification_result_list), 1)
    self.assertLessEqual(len(classification_result_list), 2)
    # Checks the first result.
    classification_result = classification_result_list[0]
    self.assertEqual(classification_result.timestamp_ms, 0)
    self.assertLen(classification_result.classifications, 2)
    # Checks the first head.
    yamnet_classifcation = classification_result.classifications[0]
    self.assertEqual(yamnet_classifcation.head_index, 0)
    self.assertEqual(yamnet_classifcation.head_name, 'yamnet_classification')
    self.assertLen(yamnet_classifcation.categories,
                   first_head_expected_num_categories)
    # Checks the second head.
    yamnet_category = yamnet_classifcation.categories[0]
    self.assertEqual(yamnet_category.index, 508)
    self.assertEqual(yamnet_category.category_name, 'Environmental noise')
    self.assertGreater(yamnet_category.score, 0.5)
    bird_classifcation = classification_result.classifications[1]
    self.assertEqual(bird_classifcation.head_index, 1)
    self.assertEqual(bird_classifcation.head_name, 'bird_classification')
    self.assertLen(bird_classifcation.categories,
                   second_head_expected_num_categories)
    bird_category = bird_classifcation.categories[0]
    self.assertEqual(bird_category.index, 4)
    self.assertEqual(bird_category.category_name, 'Chestnut-crowned Antpitta')
    self.assertGreater(bird_category.score, 0.93)
    # Checks the second result, if present.
    if len(classification_result_list) == 2:
      classification_result = classification_result_list[1]
      self.assertEqual(classification_result.timestamp_ms, 975)
      self.assertLen(classification_result.classifications, 2)
      # Checks the first head.
      yamnet_classifcation = classification_result.classifications[0]
      self.assertEqual(yamnet_classifcation.head_index, 0)
      self.assertEqual(yamnet_classifcation.head_name, 'yamnet_classification')
      self.assertLen(yamnet_classifcation.categories,
                     first_head_expected_num_categories)
      yamnet_category = yamnet_classifcation.categories[0]
      self.assertEqual(yamnet_category.index, 494)
      self.assertEqual(yamnet_category.category_name, 'Silence')
      self.assertGreater(yamnet_category.score, 0.9)
      bird_classifcation = classification_result.classifications[1]
      self.assertEqual(bird_classifcation.head_index, 1)
      self.assertEqual(bird_classifcation.head_name, 'bird_classification')
      self.assertLen(bird_classifcation.categories,
                     second_head_expected_num_categories)
      # Checks the second head.
      bird_category = bird_classifcation.categories[0]
      self.assertEqual(bird_category.index, 1)
      self.assertEqual(bird_category.category_name, 'White-breasted Wood-Wren')
      self.assertGreater(bird_category.score, 0.99)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _AudioClassifier.create_from_model_path(
        self.yamnet_model_path) as classifier:
      self.assertIsInstance(classifier, _AudioClassifier)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    with _AudioClassifier.create_from_options(
        _AudioClassifierOptions(
            base_options=_BaseOptions(
                model_asset_path=self.yamnet_model_path))) as classifier:
      self.assertIsInstance(classifier, _AudioClassifier)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite')
      options = _AudioClassifierOptions(base_options=base_options)
      _AudioClassifier.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.yamnet_model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _AudioClassifierOptions(base_options=base_options)
      classifier = _AudioClassifier.create_from_options(options)
      self.assertIsInstance(classifier, _AudioClassifier)

  @parameterized.parameters((_SPEECH_WAV_16K_MONO), (_SPEECH_WAV_48K_MONO))
  def test_classify_with_yamnet_model(self, audio_file):
    with _AudioClassifier.create_from_model_path(
        self.yamnet_model_path) as classifier:
      classification_result_list = classifier.classify(
          self._read_wav_file(audio_file))
      self._check_yamnet_result(classification_result_list)

  def test_classify_with_yamnet_model_and_inputs_at_different_sample_rates(
      self):
    with _AudioClassifier.create_from_model_path(
        self.yamnet_model_path) as classifier:
      for audio_file in [_SPEECH_WAV_16K_MONO, _SPEECH_WAV_48K_MONO]:
        classification_result_list = classifier.classify(
            self._read_wav_file(audio_file))
        self._check_yamnet_result(classification_result_list)

  @mock.patch('sounddevice.InputStream', return_value=mock.MagicMock())
  def test_create_audio_record_from_classifier_succeeds(self, _):
    # Creates AudioRecord instance using the classifier successfully.
    with _AudioClassifier.create_from_model_path(
        self.yamnet_model_path
    ) as classifier:
      self.assertIsInstance(classifier, _AudioClassifier)
      record = classifier.create_audio_record(1, 16000, 16000)
      self.assertIsInstance(record, _AudioRecord)
      self.assertEqual(record.channels, 1)
      self.assertEqual(record.sampling_rate, 16000)
      self.assertEqual(record.buffer_size, 16000)

  def test_max_result_options(self):
    with _AudioClassifier.create_from_options(
        _AudioClassifierOptions(
            base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
            max_results=1)) as classifier:
      for audio_file in [_SPEECH_WAV_16K_MONO, _SPEECH_WAV_16K_MONO]:
        classification_result_list = classifier.classify(
            self._read_wav_file(audio_file))
        self._check_yamnet_result(
            classification_result_list, expected_num_categories=1)

  def test_score_threshold_options(self):
    with _AudioClassifier.create_from_options(
        _AudioClassifierOptions(
            base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
            score_threshold=0.9)) as classifier:
      for audio_file in [_SPEECH_WAV_16K_MONO, _SPEECH_WAV_16K_MONO]:
        classification_result_list = classifier.classify(
            self._read_wav_file(audio_file))
        self._check_yamnet_result(
            classification_result_list, expected_num_categories=1)

  def test_allow_list_option(self):
    with _AudioClassifier.create_from_options(
        _AudioClassifierOptions(
            base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
            category_allowlist=['Speech'])) as classifier:
      for audio_file in [_SPEECH_WAV_16K_MONO, _SPEECH_WAV_16K_MONO]:
        classification_result_list = classifier.classify(
            self._read_wav_file(audio_file))
        self._check_yamnet_result(
            classification_result_list, expected_num_categories=1)

  def test_combined_allowlist_and_denylist(self):
    # Fails with combined allowlist and denylist
    with self.assertRaisesRegex(
        ValueError,
        r'`category_allowlist` and `category_denylist` are mutually '
        r'exclusive options.'):
      options = _AudioClassifierOptions(
          base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
          category_allowlist=['foo'],
          category_denylist=['bar'])
      with _AudioClassifier.create_from_options(options) as unused_classifier:
        pass

  @parameterized.parameters((_TWO_HEADS_WAV_16K_MONO),
                            (_TWO_HEADS_WAV_44K_MONO))
  def test_classify_with_two_heads_model_and_inputs_at_different_sample_rates(
      self, audio_file):
    with _AudioClassifier.create_from_model_path(
        self.two_heads_model_path) as classifier:
      classification_result_list = classifier.classify(
          self._read_wav_file(audio_file))
      self._check_two_heads_result(classification_result_list)

  def test_classify_with_two_heads_model(self):
    with _AudioClassifier.create_from_model_path(
        self.two_heads_model_path) as classifier:
      for audio_file in [_TWO_HEADS_WAV_16K_MONO, _TWO_HEADS_WAV_44K_MONO]:
        classification_result_list = classifier.classify(
            self._read_wav_file(audio_file))
        self._check_two_heads_result(classification_result_list)

  def test_classify_with_two_heads_model_with_max_results(self):
    with _AudioClassifier.create_from_options(
        _AudioClassifierOptions(
            base_options=_BaseOptions(
                model_asset_path=self.two_heads_model_path),
            max_results=1)) as classifier:
      for audio_file in [_TWO_HEADS_WAV_16K_MONO, _TWO_HEADS_WAV_44K_MONO]:
        classification_result_list = classifier.classify(
            self._read_wav_file(audio_file))
        self._check_two_heads_result(classification_result_list, 1, 1)

  def test_missing_sample_rate_in_audio_clips_mode(self):
    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_CLIPS)
    with self.assertRaisesRegex(ValueError,
                                r'Must provide the audio sample rate'):
      with _AudioClassifier.create_from_options(options) as classifier:
        classifier.classify(_AudioData(buffer_length=100))

  def test_missing_sample_rate_in_audio_stream_mode(self):
    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        result_callback=mock.MagicMock())
    with self.assertRaisesRegex(ValueError,
                                r'provide the audio sample rate in audio data'):
      with _AudioClassifier.create_from_options(options) as classifier:
        classifier.classify(_AudioData(buffer_length=100))

  def test_missing_result_callback(self):
    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM)
    with self.assertRaisesRegex(ValueError,
                                r'result callback must be provided'):
      with _AudioClassifier.create_from_options(options) as unused_classifier:
        pass

  def test_illegal_result_callback(self):
    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_CLIPS,
        result_callback=mock.MagicMock())
    with self.assertRaisesRegex(ValueError,
                                r'result callback should not be provided'):
      with _AudioClassifier.create_from_options(options) as unused_classifier:
        pass

  def test_calling_classify_in_audio_stream_mode(self):
    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        result_callback=mock.MagicMock())
    with _AudioClassifier.create_from_options(options) as classifier:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the audio clips mode'):
        classifier.classify(self._read_wav_file(_SPEECH_WAV_16K_MONO))

  def test_calling_classify_async_in_audio_clips_mode(self):
    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_CLIPS)
    with _AudioClassifier.create_from_options(options) as classifier:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the audio stream mode'):
        classifier.classify_async(self._read_wav_file(_SPEECH_WAV_16K_MONO), 0)

  def test_classify_async_calls_with_illegal_timestamp(self):
    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        result_callback=mock.MagicMock())
    with _AudioClassifier.create_from_options(options) as classifier:
      classifier.classify_async(self._read_wav_file(_SPEECH_WAV_16K_MONO), 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        classifier.classify_async(self._read_wav_file(_SPEECH_WAV_16K_MONO), 0)

  @parameterized.parameters((_SPEECH_WAV_16K_MONO), (_SPEECH_WAV_48K_MONO))
  def test_classify_async(self, audio_file):
    classification_result_list = []

    def save_result(result: _AudioClassifierResult, timestamp_ms: int):
      result.timestamp_ms = timestamp_ms
      classification_result_list.append(result)

    options = _AudioClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        max_results=1,
        result_callback=save_result)
    classifier = _AudioClassifier.create_from_options(options)
    audio_data_list = self._read_wav_file_as_stream(audio_file)
    for audio_data, timestamp_ms in audio_data_list:
      classifier.classify_async(audio_data, timestamp_ms)
    classifier.close()
    self._check_yamnet_result(
        classification_result_list, expected_num_categories=1)


if __name__ == '__main__':
  absltest.main()
