/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * A single transcribed segment emitted by AudioTranscriber.
 */
export declare interface TranscriptionSegment {
  /** The transcribed text for this segment. */
  transcript: string;
  /** Whether this segment is finalized (as opposed to an interim hypothesis). */
  isFinal: boolean;
  /** Start timestamp of the segment in milliseconds. */
  fromTimestampMs: number;
  /** End timestamp of the segment in milliseconds. */
  toTimestampMs: number;
}

/**
 * Result of an AudioTranscriber invocation.
 */
export declare interface AudioTranscriberResult {
  /** Ordered list of transcribed segments. */
  segments: TranscriptionSegment[];
  /** Concatenated full transcript across all segments in this result. */
  text: string;
}
