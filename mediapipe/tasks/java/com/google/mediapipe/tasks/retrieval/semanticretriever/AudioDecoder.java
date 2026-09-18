/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.retrieval.semanticretriever;

import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import com.google.mediapipe.tasks.components.containers.AudioData;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.ArrayList;

/** Stateless helper class managing the decoding of audio files from a Uri. */
final class AudioDecoder {

  private AudioDecoder() {}

  /**
   * Decodes an audio file from the specified Uri into {@link AudioData}.
   *
   * @param context the Android context.
   * @param uri the Uri of the audio file.
   * @return the decoded AudioData.
   */
  public static AudioData decodeAudio(Context context, Uri uri) {
    MediaExtractor extractor = new MediaExtractor();
    MediaCodec codec = null;
    try {
      extractor.setDataSource(context, uri, null);
      MediaFormat format = null;
      String mime = null;
      for (int i = 0; i < extractor.getTrackCount(); i++) {
        MediaFormat f = extractor.getTrackFormat(i);
        String m = f.getString(MediaFormat.KEY_MIME);
        if (m != null && m.startsWith("audio/")) {
          extractor.selectTrack(i);
          format = f;
          mime = m;
          break;
        }
      }
      if (format == null || mime == null) {
        throw new IllegalArgumentException("No audio track found in URI: " + uri);
      }

      int sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
      int channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
      codec = MediaCodec.createDecoderByType(mime);
      codec.configure(format, null, null, 0);
      codec.start();

      boolean isEos = false;
      MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
      ArrayList<float[]> pcmChunks = new ArrayList<>();
      int totalPcmLength = 0;

      while (true) {
        if (!isEos) {
          int inIndex = codec.dequeueInputBuffer(10000L);
          if (inIndex >= 0) {
            ByteBuffer buffer = codec.getInputBuffer(inIndex);
            int sampleSize = extractor.readSampleData(buffer, 0);
            if (sampleSize < 0) {
              codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
              isEos = true;
            } else {
              codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.getSampleTime(), 0);
              extractor.advance();
            }
          }
        }

        int outIndex = codec.dequeueOutputBuffer(info, 10000L);
        if (outIndex >= 0) {
          ByteBuffer buffer = codec.getOutputBuffer(outIndex);
          if (info.size > 0) {
            ShortBuffer shortBuffer = buffer.order(ByteOrder.nativeOrder()).asShortBuffer();
            float[] chunk = new float[info.size / 2];
            for (int i = 0; i < chunk.length; i++) {
              chunk[i] = shortBuffer.get(i) / 32768.0f;
            }
            pcmChunks.add(chunk);
            totalPcmLength += chunk.length;
          }
          codec.releaseOutputBuffer(outIndex, false);
          if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
            break;
          }
        }
      }

      float[] fullPcm = new float[totalPcmLength];
      int offset = 0;
      for (float[] chunk : pcmChunks) {
        System.arraycopy(chunk, 0, fullPcm, offset, chunk.length);
        offset += chunk.length;
      }

      AudioData.AudioDataFormat audioFormat =
          AudioData.AudioDataFormat.builder()
              .setNumOfChannels(channels)
              .setSampleRate(sampleRate)
              .build();
      AudioData audioData = AudioData.create(audioFormat, totalPcmLength / channels);
      audioData.load(fullPcm);
      return audioData;

    } catch (Exception e) {
      throw new RuntimeException("Failed to load audio from URI: " + uri, e);
    } finally {
      if (codec != null) {
        try {
          codec.stop();
          codec.release();
        } catch (Exception e) {
          // Ignore
        }
      }
      try {
        extractor.release();
      } catch (Exception e) {
        // Ignore
      }
    }
  }
}
