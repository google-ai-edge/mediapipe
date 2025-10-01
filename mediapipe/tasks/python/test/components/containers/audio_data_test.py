# Copyright 2025 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for AudioData conversion between Python and C."""

import ctypes
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy.io import wavfile

from mediapipe.tasks.python.components.containers import audio_data as audio_data_lib
from mediapipe.tasks.python.components.containers import audio_data_c as audio_data_c_lib
from mediapipe.tasks.python.test import test_utils

_AudioData = audio_data_lib.AudioData
_AudioDataC = audio_data_c_lib.AudioDataC
_AudioDataFormat = audio_data_lib.AudioDataFormat

_TEST_DATA_DIR = 'mediapipe/tasks/testdata/audio'
_WAV_FILE = 'speech_16000_hz_mono.wav'
_NUM_CHANNELS = 1
_SAMPLE_RATE = 16000
_BUFFER_LEN = 68360


class AudioDataTest(parameterized.TestCase):

  def _read_test_file(self) -> np.ndarray:
    """Reads the test file and returns the audio data as a NumPy array."""
    _, buffer = wavfile.read(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, _WAV_FILE))
    )
    return buffer.astype(np.float32) / np.iinfo(np.int16).max

  def _create_audio_data_from_ctypes(
      self, audio_data_c: _AudioDataC
  ) -> _AudioData:
    """Creates an `AudioData` object from a ctypes audio data object."""
    sample_rate = audio_data_c.sample_rate
    buffer = np.ctypeslib.as_array(
        audio_data_c.audio_data, shape=(audio_data_c.audio_data_size,)
    )
    return _AudioData.create_from_array(buffer, sample_rate)

  def _assert_matches_file_properties(
      self, audio_data: _AudioData, expected_buffer: np.ndarray | None = None
  ):
    """Asserts that the audio data matches the properties of the test file."""
    self.assertEqual(audio_data.audio_format.num_channels, _NUM_CHANNELS)
    self.assertEqual(audio_data.audio_format.sample_rate, _SAMPLE_RATE)
    self.assertLen(audio_data.buffer, _BUFFER_LEN)
    if expected_buffer is not None:
      self.assertTrue(np.array_equal(audio_data.buffer, expected_buffer))

  def test_create_from_array_succeeds(self):
    buffer = self._read_test_file()
    audio_data = _AudioData.create_from_array(buffer, _SAMPLE_RATE)
    self._assert_matches_file_properties(audio_data)

  def test_create_from_ctypes_succeeds(self):
    buffer = self._read_test_file()

    audio_data_c_obj = _AudioDataC(
        num_channels=_NUM_CHANNELS,
        sample_rate=_SAMPLE_RATE,
        audio_data=buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        audio_data_size=buffer.size,
    )
    audio_data_py = self._create_audio_data_from_ctypes(audio_data_c_obj)

    self._assert_matches_file_properties(audio_data_py, expected_buffer=buffer)

  def test_roundtrip_conversion_keeps_data_and_format(self):
    buffer = self._read_test_file()
    audio_data = _AudioData.create_from_array(buffer, _SAMPLE_RATE)

    audio_data_c = audio_data.to_ctypes()
    audio_data_py = self._create_audio_data_from_ctypes(audio_data_c)

    self._assert_matches_file_properties(
        audio_data_py, expected_buffer=audio_data.buffer
    )


if __name__ == '__main__':
  absltest.main()
