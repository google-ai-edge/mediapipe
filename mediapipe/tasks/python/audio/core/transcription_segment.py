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
"""Shared transcription segment type for MediaPipe Python Audio tasks."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class TranscriptionSegment:
  """A single transcribed segment with start and end timestamps.

  Attributes:
    transcript: The recognized text for this audio segment.
    is_final: Whether this segment is finalized or an interim hypothesis.
    from_timestamp_ms: Segment start time in milliseconds relative to stream
      start.
    to_timestamp_ms: Segment end time in milliseconds relative to stream start.
  """

  transcript: str
  is_final: bool
  from_timestamp_ms: int
  to_timestamp_ms: int
