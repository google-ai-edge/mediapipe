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
"""Tests for audio embedder."""
import enum
import os
from typing import List, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from scipy.io import wavfile

from mediapipe.tasks.python.audio import audio_embedder
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.audio.core import audio_task_running_mode
from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils

_AudioEmbedder = audio_embedder.AudioEmbedder
_AudioEmbedderOptions = audio_embedder.AudioEmbedderOptions
_AudioEmbedderResult = audio_embedder.AudioEmbedderResult
_AudioData = audio_data_module.AudioData
_AudioRecord = audio_record.AudioRecord
_BaseOptions = base_options_module.BaseOptions
_RUNNING_MODE = audio_task_running_mode.AudioTaskRunningMode

_YAMNET_MODEL_FILE = 'yamnet_embedding_metadata.tflite'
_YAMNET_MODEL_SAMPLE_RATE = 16000
_SPEECH_WAV_16K_MONO = 'speech_16000_hz_mono.wav'
_SPEECH_WAV_48K_MONO = 'speech_48000_hz_mono.wav'
_TWO_HEADS_WAV_16K_MONO = 'two_heads_16000_hz_mono.wav'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/audio'
_YAMNET_NUM_OF_SAMPLES = 15600
_MILLISECONDS_PER_SECOND = 1000
# Tolerance for embedding vector coordinate values.
_EPSILON = 3e-6


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class AudioEmbedderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.yamnet_model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _YAMNET_MODEL_FILE))

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

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _AudioEmbedder.create_from_model_path(
        self.yamnet_model_path) as embedder:
      self.assertIsInstance(embedder, _AudioEmbedder)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    with _AudioEmbedder.create_from_options(
        _AudioEmbedderOptions(
            base_options=_BaseOptions(
                model_asset_path=self.yamnet_model_path))) as embedder:
      self.assertIsInstance(embedder, _AudioEmbedder)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite')
      options = _AudioEmbedderOptions(base_options=base_options)
      _AudioEmbedder.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.yamnet_model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _AudioEmbedderOptions(base_options=base_options)
      embedder = _AudioEmbedder.create_from_options(options)
      self.assertIsInstance(embedder, _AudioEmbedder)

  @parameterized.parameters(
      # Same audio inputs but different sample rates.
      (False, False, ModelFileType.FILE_NAME, _SPEECH_WAV_16K_MONO,
       _SPEECH_WAV_48K_MONO, 1024, (0, 0)),
      (False, False, ModelFileType.FILE_CONTENT, _SPEECH_WAV_16K_MONO,
       _SPEECH_WAV_48K_MONO, 1024, (0, 0)))
  def test_embed_with_yamnet_model(self, l2_normalize, quantize,
                                   model_file_type, audio_file0, audio_file1,
                                   expected_size, expected_first_values):
    # Creates embedder.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.yamnet_model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.yamnet_model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _AudioEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)

    with _AudioEmbedder.create_from_options(options) as embedder:
      embedding_result0_list = embedder.embed(self._read_wav_file(audio_file0))
      embedding_result1_list = embedder.embed(self._read_wav_file(audio_file1))

      # Checks embeddings.
      expected_result0_value, expected_result1_value = expected_first_values
      self._check_embedding_size(embedding_result0_list[0], quantize,
                                 expected_size)
      self._check_embedding_size(embedding_result1_list[0], quantize,
                                 expected_size)
      self._check_embedding_value(embedding_result0_list[0],
                                  expected_result0_value)
      self._check_embedding_value(embedding_result1_list[0],
                                  expected_result1_value)
      self.assertLen(embedding_result0_list, 5)
      self.assertLen(embedding_result1_list, 5)

  @mock.patch('sounddevice.InputStream', return_value=mock.MagicMock())
  def test_create_audio_record_from_embedder_succeeds(self, _):
    # Creates AudioRecord instance using the embedder successfully.
    with _AudioEmbedder.create_from_model_path(
        self.yamnet_model_path
    ) as embedder:
      self.assertIsInstance(embedder, _AudioEmbedder)
      record = embedder.create_audio_record(1, 16000, 16000)
      self.assertIsInstance(record, _AudioRecord)
      self.assertEqual(record.channels, 1)
      self.assertEqual(record.sampling_rate, 16000)
      self.assertEqual(record.buffer_size, 16000)

  def test_embed_with_yamnet_model_and_different_inputs(self):
    with _AudioEmbedder.create_from_model_path(
        self.yamnet_model_path) as embedder:
      embedding_result0_list = embedder.embed(
          self._read_wav_file(_SPEECH_WAV_16K_MONO))
      embedding_result1_list = embedder.embed(
          self._read_wav_file(_TWO_HEADS_WAV_16K_MONO))
      self.assertLen(embedding_result0_list, 5)
      self.assertLen(embedding_result1_list, 1)

  def test_missing_sample_rate_in_audio_clips_mode(self):
    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_CLIPS)
    with self.assertRaisesRegex(ValueError,
                                r'Must provide the audio sample rate'):
      with _AudioEmbedder.create_from_options(options) as embedder:
        embedder.embed(_AudioData(buffer_length=100))

  def test_missing_sample_rate_in_audio_stream_mode(self):
    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        result_callback=mock.MagicMock())
    with self.assertRaisesRegex(ValueError,
                                r'provide the audio sample rate in audio data'):
      with _AudioEmbedder.create_from_options(options) as embedder:
        embedder.embed(_AudioData(buffer_length=100))

  def test_missing_result_callback(self):
    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM)
    with self.assertRaisesRegex(ValueError,
                                r'result callback must be provided'):
      with _AudioEmbedder.create_from_options(options) as unused_embedder:
        pass

  def test_illegal_result_callback(self):
    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_CLIPS,
        result_callback=mock.MagicMock())
    with self.assertRaisesRegex(ValueError,
                                r'result callback should not be provided'):
      with _AudioEmbedder.create_from_options(options) as unused_embedder:
        pass

  def test_calling_embed_in_audio_stream_mode(self):
    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        result_callback=mock.MagicMock())
    with _AudioEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the audio clips mode'):
        embedder.embed(self._read_wav_file(_SPEECH_WAV_16K_MONO))

  def test_calling_embed_async_in_audio_clips_mode(self):
    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_CLIPS)
    with _AudioEmbedder.create_from_options(options) as embedder:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the audio stream mode'):
        embedder.embed_async(self._read_wav_file(_SPEECH_WAV_16K_MONO), 0)

  def test_embed_async_calls_with_illegal_timestamp(self):
    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        result_callback=mock.MagicMock())
    with _AudioEmbedder.create_from_options(options) as embedder:
      embedder.embed_async(self._read_wav_file(_SPEECH_WAV_16K_MONO), 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        embedder.embed_async(self._read_wav_file(_SPEECH_WAV_16K_MONO), 0)

  @parameterized.parameters(
      # Same audio inputs but different sample rates.
      (False, False, _SPEECH_WAV_16K_MONO, _SPEECH_WAV_48K_MONO))
  def test_embed_async(self, l2_normalize, quantize, audio_file0, audio_file1):
    embedding_result_list = []
    embedding_result_list_copy = embedding_result_list.copy()

    def save_result(result: _AudioEmbedderResult, timestamp_ms: int):
      result.timestamp_ms = timestamp_ms
      embedding_result_list.append(result)

    options = _AudioEmbedderOptions(
        base_options=_BaseOptions(model_asset_path=self.yamnet_model_path),
        running_mode=_RUNNING_MODE.AUDIO_STREAM,
        l2_normalize=l2_normalize,
        quantize=quantize,
        result_callback=save_result)

    with _AudioEmbedder.create_from_options(options) as embedder:
      audio_data0_list = self._read_wav_file_as_stream(audio_file0)
      for audio_data, timestamp_ms in audio_data0_list:
        embedder.embed_async(audio_data, timestamp_ms)
      embedding_result0_list = embedding_result_list

    with _AudioEmbedder.create_from_options(options) as embedder:
      audio_data1_list = self._read_wav_file_as_stream(audio_file1)
      embedding_result_list = embedding_result_list_copy
      for audio_data, timestamp_ms in audio_data1_list:
        embedder.embed_async(audio_data, timestamp_ms)
      embedding_result1_list = embedding_result_list

    self.assertLen(embedding_result0_list, 5)
    self.assertLen(embedding_result1_list, 5)


if __name__ == '__main__':
  absltest.main()
