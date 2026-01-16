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
"""The running mode of MediaPipe Audio Tasks."""

import enum
import types

from mediapipe.tasks.python.core.optional_dependencies import doc_controls

# The C API constants that map to the Python enum values.
_CTYPE_VALUE_MAP = types.MappingProxyType({
    'AUDIO_CLIPS': 1,
    'AUDIO_STREAM': 2,
})


class AudioTaskRunningMode(enum.Enum):
  """MediaPipe audio task running mode.

  Attributes:
    AUDIO_CLIPS: The mode for running a mediapipe audio task on independent
      audio clips.
    AUDIO_STREAM: The mode for running a mediapipe audio task on an audio
      stream, such as from microphone.
  """
  AUDIO_CLIPS = 'AUDIO_CLIPS'
  AUDIO_STREAM = 'AUDIO_STREAM'

  @property
  @doc_controls.do_not_generate_docs
  def ctype(self) -> int:
    """Generates a C API int object."""
    ctype_value = _CTYPE_VALUE_MAP.get(self.value)
    assert (
        ctype_value is not None
    ), f'Unsupported audio task running mode: {self}'
    return ctype_value
