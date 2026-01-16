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
"""MediaPipe audio data."""

import ctypes
import dataclasses
from typing import Optional

import numpy as np

from mediapipe.tasks.python.components.containers import audio_data_c
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class AudioDataFormat:
  """Audio format metadata.

  Attributes:
    num_channels: the number of channels of the audio data.
    sample_rate: the audio sample rate.
  """
  num_channels: int = 1
  sample_rate: Optional[float] = None


class AudioData(object):
  """MediaPipe Tasks' audio container."""

  def __init__(
      self,
      buffer_length: int,
      audio_format: AudioDataFormat = AudioDataFormat()
  ) -> None:
    """Initializes the `AudioData` object.

    Args:
      buffer_length: the length of the audio buffer.
      audio_format: the audio format metadata.
    """
    self._audio_format = audio_format
    self._buffer = np.zeros([buffer_length, self._audio_format.num_channels],
                            dtype=np.float32)

  def clear(self):
    """Clears the internal buffer and fill it with zeros."""
    self._buffer.fill(0)

  def load_from_array(self,
                      src: np.ndarray,
                      offset: int = 0,
                      size: int = -1) -> None:
    """Loads the audio data from a NumPy array.

    Args:
      src: A NumPy source array contains the input audio.
      offset: An optional offset for loading a slice of the `src` array to the
        buffer.
      size: An optional size parameter denoting the number of samples to load
        from the `src` array.

    Raises:
      ValueError: If the input array has an incorrect shape or if
        `offset` + `size` exceeds the length of the `src` array.
    """
    if len(src.shape) == 1:
      if self._audio_format.num_channels != 1:
        raise ValueError(f"Input audio is mono, but the audio data is expected "
                         f"to have {self._audio_format.num_channels} channels.")
    elif src.shape[1] != self._audio_format.num_channels:
      raise ValueError(f"Input audio contains an invalid number of channels. "
                       f"Expect {self._audio_format.num_channels}.")

    if size < 0:
      size = len(src)

    if offset + size > len(src):
      raise ValueError(
          f"Index out of range. offset {offset} + size {size} should be <= "
          f"src's length: {len(src)}")

    if len(src) >= len(self._buffer):
      # If the internal buffer is shorter than the load target (src), copy
      # values from the end of the src array to the internal buffer.
      new_offset = offset + size - len(self._buffer)
      new_size = len(self._buffer)
      self._buffer = src[new_offset:new_offset + new_size].copy()
    else:
      # Shift the internal buffer backward and add the incoming data to the end
      # of the buffer.
      shift = size
      self._buffer = np.roll(self._buffer, -shift, axis=0)
      self._buffer[-shift:, :] = src[offset:offset + size].copy()

  @classmethod
  def create_from_array(cls,
                        src: np.ndarray,
                        sample_rate: Optional[float] = None) -> "AudioData":
    """Creates an `AudioData` object from a NumPy array.

    Args:
      src: A NumPy source array contains the input audio.
      sample_rate: the optional audio sample rate.

    Returns:
      An `AudioData` object that contains a copy of the NumPy source array as
      the data.
    """
    src = src.astype(np.float32)
    obj = cls(
        buffer_length=src.shape[0],
        audio_format=AudioDataFormat(
            num_channels=1 if len(src.shape) == 1 else src.shape[1],
            sample_rate=sample_rate))
    obj.load_from_array(src)
    return obj

  @property
  def audio_format(self) -> AudioDataFormat:
    """Gets the audio format of the audio."""
    return self._audio_format

  @property
  def buffer_length(self) -> int:
    """Gets the sample count of the audio."""
    return self._buffer.shape[0]

  @property
  def buffer(self) -> np.ndarray:
    """Gets the internal buffer."""
    return self._buffer

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> audio_data_c.AudioDataC:
    """Converts the object to a ctypes audio data object."""
    return audio_data_c.AudioDataC(
        num_channels=self.audio_format.num_channels,
        sample_rate=self.audio_format.sample_rate,
        audio_data=self._buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        audio_data_size=self._buffer.size,
    )
