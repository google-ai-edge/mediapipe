# Copyright 2026 The MediaPipe Authors.
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
"""Unit tests for MediaPipe TextToSpeech Python task."""

import ctypes
from unittest import mock

from absl.testing import absltest
import numpy as np

from mediapipe.tasks.python.audio import text_to_speech
from mediapipe.tasks.python.core import base_options

_BaseOptions = base_options.BaseOptions
_TextToSpeech = text_to_speech.TextToSpeech
_TextToSpeechOptions = text_to_speech.TextToSpeechOptions
_MpTtsVoiceInfoC = text_to_speech._MpTtsVoiceInfoC
_SYNTHETIC_MODEL = ':synthetic_test_model:'


class TextToSpeechTest(absltest.TestCase):

  def test_synthesize_unary_and_list_voices_precondition(self):
    with _TextToSpeech.create_from_model_path(_SYNTHETIC_MODEL) as tts:
      with self.assertRaises(RuntimeError):
        tts.list_voices()

      res = tts.synthesize('Hello from MediaPipe Python unit tests.')
      self.assertEqual(res.samples.dtype, np.float32)
      self.assertGreater(res.samples.size, 0)
      self.assertEqual(res.sample_rate_hz, 24000)
      self.assertGreater(res.duration_seconds, 0.0)

  def test_list_voices_marshaling(self):
    with _TextToSpeech.create_from_model_path(_SYNTHETIC_MODEL) as tts:
      voice_c = _MpTtsVoiceInfoC(
          voice_id=b'af_heart',
          display_name=b'Heart',
          locale=b'en-US',
          sample_rate_hz=24000,
          is_downloaded=True,
      )

      def fake_list_voices(handle, list_ptr, err_ptr):
        del handle, err_ptr
        list_ptr._obj.voices = ctypes.pointer(voice_c)
        list_ptr._obj.num_voices = 1
        return 0

      with (
          mock.patch.object(
              tts._lib, 'MpTextToSpeechListVoices', side_effect=fake_list_voices
          ),
          mock.patch.object(tts._lib, 'MpTextToSpeechVoiceListFree'),
      ):
        voices = tts.list_voices()
        self.assertLen(voices, 1)
        self.assertEqual(voices[0].voice_id, 'af_heart')
        self.assertEqual(voices[0].display_name, 'Heart')
        self.assertEqual(voices[0].locale, 'en-US')
        self.assertEqual(voices[0].sample_rate_hz, 24000)
        self.assertTrue(voices[0].is_downloaded)

  def test_synthesize_stream(self):
    opts = _TextToSpeechOptions(
        base_options=_BaseOptions(model_asset_path=_SYNTHETIC_MODEL),
        default_voice_id='af_heart',
        default_locale='en-US',
        speaking_rate=1.0,
    )
    with _TextToSpeech.create_from_options(opts) as tts:
      chunks = list(
          tts.synthesize_stream(
              'First sentence for streaming. Second sentence for streaming.',
              voice_id='af_heart',
              locale='en-US',
          )
      )
      self.assertNotEmpty(chunks)
      total_samples = sum(c.samples.size for c in chunks)
      self.assertGreater(total_samples, 0)
      self.assertTrue(chunks[-1].is_final)
      tts.cancel()

  def test_invalid_model_path_raises_runtime_error(self):
    with self.assertRaises(RuntimeError):
      _TextToSpeech.create_from_model_path('/nonexistent/tts.litertlm')

  def test_edge_cases_and_status_handling(self):
    empty_res = text_to_speech.TtsAudioResult(
        samples=np.zeros(0, dtype=np.float32), sample_rate_hz=0
    )
    self.assertEqual(empty_res.duration_seconds, 0.0)
    with self.assertRaisesRegex(RuntimeError, 'MpStatus=5'):
      text_to_speech._check_status(
          mock.MagicMock(), 5, ctypes.c_char_p(None)
      )
    opts = _TextToSpeechOptions(
        base_options=_BaseOptions(model_asset_path=_SYNTHETIC_MODEL),
        default_voice_id='',
        default_locale='',
        cache_directory='/tmp',
    )
    with _TextToSpeech.create_from_options(opts) as tts:
      c_empty = text_to_speech._MpTtsAudioResultC(
          samples=None, num_samples=0, sample_rate_hz=24000
      )
      with mock.patch.object(tts._lib, 'MpTextToSpeechFreeAudioResult'):
        extracted = tts._extract_audio_result(c_empty)
        self.assertEqual(extracted.samples.size, 0)


if __name__ == '__main__':
  absltest.main()
