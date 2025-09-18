# Copyright 2025 The MediaPipe Authors.
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
"""MediaPipe Keypoint C API types."""

import ctypes

from mediapipe.tasks.python.components.containers import keypoint as keypoint_lib


class NormalizedKeypointC(ctypes.Structure):
  """A keypoint in normalized coordinates."""

  _fields_ = [
      ('x', ctypes.c_float),
      ('y', ctypes.c_float),
      ('label', ctypes.c_char_p),
      ('score', ctypes.c_float),
  ]

  def to_python_normalized_keypoint(self) -> keypoint_lib.NormalizedKeypoint:
    """Converts a ctypes NormalizedKeypointC to a NormalizedKeypoint object."""
    return keypoint_lib.NormalizedKeypoint(
        x=self.x,
        y=self.y,
        label=self.label.decode('utf-8') if self.label else None,
        score=self.score,
    )
