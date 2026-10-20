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
"""Unit tests for MediaPipe AudioTranscriber Python task."""

import ctypes
from unittest import mock

from absl.testing import absltest
import numpy as np

from mediapipe.tasks.python.audio import audio_transcriber
from mediapipe.tasks.python.core import base_options

_BaseOptions = base_options.BaseOptions
_AudioTranscriber = audio_transcriber.AudioTranscriber
_AudioTranscriberOptions = audio_transcriber.AudioTranscriberOptions
_ModelArchitecture = audio_transcriber.ModelArchitecture
_MpTranscriptionSegmentC = audio_transcriber._MpTranscriptionSegmentC


class _FakeAudioTranscriberLib:
  """Fake ctypes CDLL verifying C struct marshaling and memory contiguity."""

  def __init__(self):
    self.last_samples: list[float] = []
    self.last_end_of_stream: bool = False
    self.prewarmed: bool = False
    self.closed: bool = False
    self._seg = _MpTranscriptionSegmentC(
        transcript=b'hello world',
        is_final=True,
        from_timestamp_ms=0,
        to_timestamp_ms=1000,
    )
    for name, fn in (
        ('MpAudioTranscriberCreate', self._create),
        ('MpAudioTranscriberPrewarm', self._prewarm),
        ('MpAudioTranscriberTranscribe', self._transcribe),
        ('MpAudioTranscriberFinishStream', self._finish_stream),
        ('MpAudioTranscriberCloseResult', self._close_result),
        ('MpAudioTranscriberClose', self._close),
        ('MpErrorFree', self._error_free),
    ):
      setattr(self, name, fn)

  def _create(self, opts_ptr, handle_ptr, err_ptr):
    del opts_ptr, err_ptr
    handle_ptr._obj.value = 0x1234
    return 0

  def _prewarm(self, handle, err_ptr):
    del handle, err_ptr
    self.prewarmed = True
    return 0

  def _transcribe(
      self, handle, samples_ptr, num_samples, end_of_stream, res_ptr, err_ptr
  ):
    del handle, err_ptr
    self.last_samples = [float(samples_ptr[i]) for i in range(num_samples)]
    self.last_end_of_stream = bool(end_of_stream)
    res_ptr._obj.segments = ctypes.pointer(self._seg)
    res_ptr._obj.segments_count = 1
    return 0

  def _finish_stream(self, handle, res_ptr, err_ptr):
    del handle, err_ptr
    res_ptr._obj.segments = ctypes.pointer(self._seg)
    res_ptr._obj.segments_count = 1
    return 0

  def _close_result(self, res_ptr):
    del res_ptr

  def _close(self, handle, err_ptr):
    del handle, err_ptr
    self.closed = True
    return 0

  def _error_free(self, err_ptr):
    del err_ptr


class AudioTranscriberTest(absltest.TestCase):

  def test_create_prewarm_and_transcribe_unary(self):
    fake_lib = _FakeAudioTranscriberLib()
    with mock.patch.object(
        audio_transcriber, '_get_lib', return_value=fake_lib
    ):
      with _AudioTranscriber.create_from_model_path('test.litertlm') as asr:
        asr.prewarm()
        self.assertTrue(fake_lib.prewarmed)
        result = asr.transcribe([0.25, -0.5, 0.75], end_of_stream=True)
        self.assertLen(result.segments, 1)
        self.assertEqual(result.text, 'hello world')
        self.assertTrue(result.segments[0].is_final)
        self.assertEqual(result.segments[0].from_timestamp_ms, 0)
        self.assertEqual(result.segments[0].to_timestamp_ms, 1000)
        np.testing.assert_allclose(fake_lib.last_samples, [0.25, -0.5, 0.75])
      self.assertTrue(fake_lib.closed)

  def test_transcribe_strided_non_contiguous_array(self):
    fake_lib = _FakeAudioTranscriberLib()
    full_audio = np.array([1.0, 99.0, 2.0, 99.0, 3.0, 99.0], dtype=np.float32)
    strided_audio = full_audio[::2]
    self.assertFalse(strided_audio.flags['C_CONTIGUOUS'])
    with mock.patch.object(
        audio_transcriber, '_get_lib', return_value=fake_lib
    ):
      with _AudioTranscriber.create_from_model_path('test.litertlm') as asr:
        result = asr.transcribe(strided_audio, end_of_stream=True)
        self.assertEqual(result.text, 'hello world')
        np.testing.assert_allclose(fake_lib.last_samples, [1.0, 2.0, 3.0])

  def test_streaming_transcribe_and_finish_stream(self):
    fake_lib = _FakeAudioTranscriberLib()
    opts = _AudioTranscriberOptions(
        base_options=_BaseOptions(model_asset_path='test.litertlm'),
        sample_rate_hz=16000,
        enable_interim_results=True,
        model_architecture=_ModelArchitecture.OMNI,
        prompt='Transcribe speech',
        cache_dir='/tmp',
    )
    with mock.patch.object(
        audio_transcriber, '_get_lib', return_value=fake_lib
    ):
      with _AudioTranscriber.create_from_options(opts) as asr:
        partial = asr.transcribe([0.1, 0.2], end_of_stream=False)
        self.assertFalse(fake_lib.last_end_of_stream)
        final = asr.finish_stream()
        self.assertEqual(partial.text, 'hello world')
        self.assertEqual(final.text, 'hello world')

  def test_invalid_model_path_raises_runtime_error_in_native_c_lib(self):
    with self.assertRaises(RuntimeError):
      _AudioTranscriber.create_from_model_path('/nonexistent/model.litertlm')

  def test_invalid_sample_rate_raises_runtime_error_in_native_c_lib(self):
    opts = _AudioTranscriberOptions(
        base_options=_BaseOptions(model_asset_path='/tmp/fake.litertlm'),
        sample_rate_hz=-1,
    )
    with self.assertRaises(RuntimeError):
      _AudioTranscriber.create_from_options(opts)

  def test_check_status_and_empty_segment_handling(self):
    fake_lib = _FakeAudioTranscriberLib()
    with self.assertRaisesRegex(RuntimeError, 'MpStatus=3'):
      audio_transcriber._check_status(fake_lib, 3, ctypes.c_char_p(None))
    empty_seg = _MpTranscriptionSegmentC(
        transcript=None,
        is_final=False,
        from_timestamp_ms=10,
        to_timestamp_ms=20,
    )
    fake_lib._seg = empty_seg
    with mock.patch.object(
        audio_transcriber, '_get_lib', return_value=fake_lib
    ):
      with _AudioTranscriber.create_from_model_path('test.litertlm') as asr:
        result = asr.transcribe([0.0], end_of_stream=True)
        self.assertLen(result.segments, 1)
        self.assertEqual(result.segments[0].transcript, '')
        self.assertEqual(result.text, '')


if __name__ == '__main__':
  absltest.main()
