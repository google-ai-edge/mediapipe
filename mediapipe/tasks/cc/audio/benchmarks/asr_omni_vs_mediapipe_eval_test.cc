/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/tasks/cc/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/cc/audio/benchmarks/eval_benchmark_utils.h"
#include "odml/audio/transcription/speech_processor.h"    // from @odml
#include "odml/audio/transcription/testing/test_utils.h"  // from @odml
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe::tasks::audio::benchmarks {
namespace {

using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriber;
using ::odml::audio::transcription::SpeechProcessor;
using ::odml::audio::transcription::testing::kAudio123456789;
using ::odml::audio::transcription::testing::kAudioCarolineAndPeter;
using ::odml::audio::transcription::testing::kAudioHelloIsThereAnybodyInThere;
using ::odml::audio::transcription::testing::kAudioHelloWhatIsGoingOn;
using ::odml::audio::transcription::testing::kAudioShouldntMissTheBeginning;
using ::odml::audio::transcription::testing::kAudioThereIsSomethingInTheWater;
using ::odml::audio::transcription::testing::kAudioWhatColorIsCheese;
using ::odml::audio::transcription::testing::kAudioWhisper;
using ::odml::audio::transcription::testing::kAudioWikipedia;
using ::odml::audio::transcription::testing::ReadWavFile;
using ::testing::Gt;
using ::testing::HasSubstr;

constexpr int kStreamChunk500Ms = 500;
constexpr int kMinWikipediaSeconds = 30;
constexpr int kMsPerSecond = 1000;

struct AsrModelPair {
  std::unique_ptr<SpeechProcessor> omni_16k;
  std::unique_ptr<SpeechProcessor> omni_48k;
  std::unique_ptr<AudioTranscriber> mp_16k;
  std::unique_ptr<AudioTranscriber> mp_48k;
  double omni_init_ms = 0.0;
  double mp_init_ms = 0.0;
  int omni_thread_delta = 0;
  int mp_thread_delta = 0;

  SpeechProcessor& SelectOmni(int sample_rate_hz) {
    return (sample_rate_hz == kSampleRate48kHz) ? *omni_48k : *omni_16k;
  }
  AudioTranscriber& SelectMediaPipe(int sample_rate_hz) {
    return (sample_rate_hz == kSampleRate48kHz) ? *mp_48k : *mp_16k;
  }
};

std::vector<AsrEvalCase> GetStandardAsrEvalCases() {
  return {
      {kAudioWhatColorIsCheese, "what color is cheese", "what color is cheese"},
      {kAudioShouldntMissTheBeginning, "beginning",
       "its not really missing the beginning"},
      {kAudioHelloWhatIsGoingOn, "what is going on", "hello what is going on"},
      {kAudioThereIsSomethingInTheWater, "something in the water",
       "there is something in the water"},
      {kAudioWhisper, "whisper", "whisper"},
      {kAudio123456789, "one two three",
       "one two three four five six seven eight nine"},
      {kAudioHelloIsThereAnybodyInThere, "anybody in there",
       "hello is there anybody in there"},
      {kAudioCarolineAndPeter, "caroline", "caroline and peter"},
  };
}

absl::StatusOr<AsrModelPair> CreateAsrModelPair(
    absl::string_view model_filename) {
  ABSL_ASSIGN_OR_RETURN(std::string model_path,
                        ResolveStagedModel(model_filename));
  AsrModelPair pair;

  int threads_before_omni = GetCurrentProcessThreadCount();
  absl::Time t_omni_init = absl::Now();
  ABSL_ASSIGN_OR_RETURN(pair.omni_16k,
                        CreatePrewarmedOmniAsr(model_path, kSampleRate16kHz));
  pair.omni_init_ms = absl::ToDoubleMilliseconds(absl::Now() - t_omni_init);
  pair.omni_thread_delta =
      std::max(0, GetCurrentProcessThreadCount() - threads_before_omni);
  ABSL_ASSIGN_OR_RETURN(pair.omni_48k,
                        CreatePrewarmedOmniAsr(model_path, kSampleRate48kHz));

  int threads_before_mp = GetCurrentProcessThreadCount();
  absl::Time t_mp_init = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      pair.mp_16k, CreatePrewarmedMediaPipeAsr(model_path, kSampleRate16kHz));
  pair.mp_init_ms = absl::ToDoubleMilliseconds(absl::Now() - t_mp_init);
  pair.mp_thread_delta =
      std::max(0, GetCurrentProcessThreadCount() - threads_before_mp);
  ABSL_ASSIGN_OR_RETURN(
      pair.mp_48k, CreatePrewarmedMediaPipeAsr(model_path, kSampleRate48kHz));
  return pair;
}

void VerifyShortAndMediumWavParity(absl::string_view model_filename,
                                   AsrModelPair& pair, int mode_ms) {
  const std::string mode_label = (mode_ms == 0) ? "Unary" : "Stream-500ms";
  const std::vector<AsrEvalCase> eval_cases = GetStandardAsrEvalCases();
  int exact_matches = 0;

  for (const auto& tc : eval_cases) {
    const auto wav = ReadWavFile(tc.wav_name);
    SpeechProcessor& omni_proc = pair.SelectOmni(wav.sample_rate_hz);
    AudioTranscriber& mp_transcriber = pair.SelectMediaPipe(wav.sample_rate_hz);
    const size_t chunk_samples =
        (mode_ms == 0) ? 0
                       : static_cast<size_t>((mode_ms * wav.sample_rate_hz) /
                                             kMsPerSecond);

    ABSL_MP_ASSERT_OK_AND_ASSIGN(AsrRunMetrics omni_m,
                                 RunOmniAsr(omni_proc, wav.audio_samples,
                                            wav.sample_rate_hz, chunk_samples));
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics mp_m, RunMediaPipeAsr(mp_transcriber, wav.audio_samples,
                                            wav.sample_rate_hz, chunk_samples));

    std::string norm_omni = NormalizeTranscript(omni_m.final_transcript);
    std::string norm_mp = NormalizeTranscript(mp_m.final_transcript);
    EXPECT_THAT(norm_mp, HasSubstr(tc.expected_substr))
        << "Model=" << model_filename << " Mode=" << mode_label
        << " File=" << tc.wav_name;
    EXPECT_EQ(norm_mp, norm_omni)
        << "Transcript mismatch on " << tc.wav_name << " in " << mode_label;
    if (norm_mp == norm_omni) {
      ++exact_matches;
    }

    std::cout << absl::StrFormat(
        "  [%-12s] %-38s | Omni: %6.1f ms (RTF %.3f) | MP: %6.1f ms "
        "(RTF %.3f) | Parity: %s | \"%s\"\n",
        mode_label, tc.wav_name, omni_m.total_latency_ms, omni_m.rtf,
        mp_m.total_latency_ms, mp_m.rtf,
        (norm_mp == norm_omni ? "EXACT" : "DIFF"), norm_mp);
  }
  EXPECT_EQ(exact_matches, static_cast<int>(eval_cases.size()));
}

void VerifyWikipediaLongFormStreamingParity(AsrModelPair& pair) {
  const auto wiki_wav = ReadWavFile(kAudioWikipedia);
  ASSERT_THAT(
      wiki_wav.audio_samples.size(),
      Gt(static_cast<size_t>(kMinWikipediaSeconds * wiki_wav.sample_rate_hz)));
  SpeechProcessor& wiki_omni_proc = pair.SelectOmni(wiki_wav.sample_rate_hz);
  AudioTranscriber& wiki_mp_transcriber =
      pair.SelectMediaPipe(wiki_wav.sample_rate_hz);

  for (int chunk_ms : {100, 500, 1000}) {
    const size_t chunk_samples = static_cast<size_t>(
        (chunk_ms * wiki_wav.sample_rate_hz) / kMsPerSecond);
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics omni_wiki,
        RunOmniAsr(wiki_omni_proc, wiki_wav.audio_samples,
                   wiki_wav.sample_rate_hz, chunk_samples));
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics mp_wiki,
        RunMediaPipeAsr(wiki_mp_transcriber, wiki_wav.audio_samples,
                        wiki_wav.sample_rate_hz, chunk_samples));

    std::string norm_omni = NormalizeTranscript(omni_wiki.final_transcript);
    std::string norm_mp = NormalizeTranscript(mp_wiki.final_transcript);
    EXPECT_THAT(norm_mp, HasSubstr("wikipedia"));
    EXPECT_EQ(norm_mp, norm_omni);
    EXPECT_EQ(mp_wiki.interim_updates, omni_wiki.interim_updates);

    std::cout << absl::StrFormat(
        "  [Wiki-34s %4dms] Omni: total=%6.1f ms, p50=%5.1f ms, p90=%5.1f "
        "ms, RTF=%.3f, interims=%d | MP: total=%6.1f ms, p50=%5.1f ms, "
        "p90=%5.1f ms, RTF=%.3f, interims=%d | Parity: %s\n",
        chunk_ms, omni_wiki.total_latency_ms, omni_wiki.p50_chunk_ms,
        omni_wiki.p90_chunk_ms, omni_wiki.rtf, omni_wiki.interim_updates,
        mp_wiki.total_latency_ms, mp_wiki.p50_chunk_ms, mp_wiki.p90_chunk_ms,
        mp_wiki.rtf, mp_wiki.interim_updates,
        (norm_mp == norm_omni ? "EXACT" : "DIFF"));
  }
}

void RunParakeetModelComprehensiveEval(absl::string_view model_filename) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(AsrModelPair pair,
                               CreateAsrModelPair(model_filename));
  std::cout << absl::StrFormat(
      "\n--- Model: %s ---\n"
      "Init Latency: Omni = %.1f ms (+%d threads) | MediaPipe = %.1f ms "
      "(+%d threads)\n",
      model_filename, pair.omni_init_ms, pair.omni_thread_delta,
      pair.mp_init_ms, pair.mp_thread_delta);

  VerifyShortAndMediumWavParity(model_filename, pair, /*mode_ms=*/0);
  VerifyShortAndMediumWavParity(model_filename, pair, kStreamChunk500Ms);
  VerifyWikipediaLongFormStreamingParity(pair);

  ABSL_MP_ASSERT_OK(pair.mp_16k->Close());
  ABSL_MP_ASSERT_OK(pair.mp_48k->Close());
}

TEST(AsrOmniVsMediaPipeEvalTest,
     ParakeetTdtOmniVsMediaPipeRealAudioAndStreamingParity) {
  RunParakeetModelComprehensiveEval(kParakeetTdtModelName);
}

TEST(AsrOmniVsMediaPipeEvalTest, ParakeetCtcOmniVsMediaPipeRealAudioParity) {
  RunParakeetModelComprehensiveEval(kParakeetCtcModelName);
}

struct OpenAsrFamilyDescriptor {
  absl::string_view model_filename;
  absl::string_view license;
  absl::string_view family_label;
};

void EvaluateOpenAsrFamilyModel(const OpenAsrFamilyDescriptor& desc) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(AsrModelPair pair,
                               CreateAsrModelPair(desc.model_filename));
  std::cout << absl::StrFormat(
      "\n--- Open ASR Model: %s (%s) | Family=%s ---\n"
      "Init Latency: Omni = %.1f ms (+%d threads) | MediaPipe = %.1f ms "
      "(+%d threads)\n",
      desc.model_filename, desc.license, desc.family_label, pair.omni_init_ms,
      pair.omni_thread_delta, pair.mp_init_ms, pair.mp_thread_delta);

  const std::vector<AsrEvalCase> subset = {
      {kAudioWhatColorIsCheese, "what color", "what color is cheese"},
      {kAudio123456789, "", "one two three four five six seven eight nine"},
      {kAudioHelloIsThereAnybodyInThere, "hello",
       "hello is there anybody in there"},
  };
  for (const auto& tc : subset) {
    const auto wav = ReadWavFile(tc.wav_name);
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics omni_m,
        RunOmniAsr(pair.SelectOmni(wav.sample_rate_hz), wav.audio_samples,
                   wav.sample_rate_hz, /*chunk_samples=*/0));
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics mp_m,
        RunMediaPipeAsr(pair.SelectMediaPipe(wav.sample_rate_hz),
                        wav.audio_samples, wav.sample_rate_hz,
                        /*chunk_samples=*/0));
    std::string norm_omni = NormalizeTranscript(omni_m.final_transcript);
    std::string norm_mp = NormalizeTranscript(mp_m.final_transcript);
    EXPECT_FALSE(norm_mp.empty()) << desc.model_filename;
    if (!tc.expected_substr.empty()) {
      EXPECT_THAT(norm_mp, HasSubstr(tc.expected_substr))
          << desc.model_filename;
    }
    EXPECT_EQ(norm_mp, norm_omni) << desc.model_filename << " " << tc.wav_name;
    std::cout << absl::StrFormat(
        "  [%-24s] %-34s | Omni: %6.1f ms (RTF %.3f) | MP: %6.1f ms "
        "(RTF %.3f) | \"%s\"\n",
        desc.model_filename, tc.wav_name, omni_m.total_latency_ms, omni_m.rtf,
        mp_m.total_latency_ms, mp_m.rtf, norm_mp);
  }
  ABSL_MP_ASSERT_OK(pair.mp_16k->Close());
  ABSL_MP_ASSERT_OK(pair.mp_48k->Close());
}

TEST(AsrOmniVsMediaPipeEvalTest,
     OpenAsrModelsArchitectureFamiliesOmniVsMediaPipeParity) {
  const std::vector<OpenAsrFamilyDescriptor> open_models = {
      {kWhisperTinyModelName, "MIT", "kSeq2SeqStateless"},
      {kMoonshineTinyModelName, "MIT", "kSeq2SeqStateless"},
      {kQwen3AsrModelName, "Apache-2.0", "kAudioLmAutoregressive"},
      {kTinyGemmaAsrModelName, "Gemma", "kAudioLmEngine"},
  };
  for (const auto& desc : open_models) {
    EvaluateOpenAsrFamilyModel(desc);
  }
}

}  // namespace
}  // namespace mediapipe::tasks::audio::benchmarks
