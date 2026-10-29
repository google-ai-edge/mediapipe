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
 * Voice metadata descriptor returned by `TextToSpeech.listVoices()`.
 */
export declare interface VoiceInfo {
  /** Unique voice identifier (e.g. "af_heart", "bf_emma", "jf_alpha"). */
  voiceId: string;
  /** Human-readable display name for the voice. */
  displayName: string;
  /** BCP-47 locale tag for the voice (e.g. "en-US", "en-GB", "ja-JP"). */
  locale: string;
  /** Output PCM audio sample rate in Hz (e.g. 24000). */
  sampleRateHz: number;
  /** Whether the voice embedding is bundled and ready for offline use. */
  isDownloaded: boolean;
}

/**
 * Word-level timing alignment mapping an input word to audio seconds.
 */
export declare interface WordTimestamp {
  /** Original surface word token from the input text. */
  word: string;
  /** Zero-based byte offset of the start of `word` in the input text. */
  startChar: number;
  /** Zero-based byte offset of the end of `word` (exclusive) in the input text. */
  endChar: number;
  /** Estimated start time in seconds within the synthesized audio waveform. */
  startSec: number;
  /** Estimated end time in seconds within the synthesized audio waveform. */
  endSec: number;
}

/**
 * Result of a unary `TextToSpeech.synthesize()` invocation.
 */
export declare interface TextToSpeechResult {
  /** Synthesized 32-bit mono float PCM audio samples in `[-1.0, 1.0]`. */
  samples: Float32Array;
  /** Audio sample rate in Hz (e.g. 24000). */
  sampleRateHz: number;
  /** Optional word-level timing alignments for read-along highlighting. */
  wordTimestamps?: WordTimestamp[];
}

/**
 * A single streaming PCM audio chunk emitted by `TextToSpeechStream`.
 */
export declare interface TextToSpeechAudioChunk {
  /** Synthesized 32-bit mono float PCM audio samples in `[-1.0, 1.0]`. */
  samples: Float32Array;
  /** Audio sample rate in Hz (e.g. 24000). */
  sampleRateHz: number;
  /** 0-based index of this chunk within the streaming session. */
  chunkIndex: number;
  /** Whether this is the final chunk in the stream. */
  isFinal: boolean;
  /** Optional word-level timing alignments for this chunk. */
  wordTimestamps?: WordTimestamp[];
}
