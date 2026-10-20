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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/tasks/cc/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/cc/audio/benchmarks/eval_benchmark_utils.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/text_to_speech.h"
#include "odml/audio/transcription/speech_processor.h"  // from @odml
#include "omni/omni_engine.h"                           // from @litert_lm
#include "runtime/framework/threadpool.h"               // from @litert_lm
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe::tasks::audio::benchmarks {
namespace {

using ::litert::omni::OmniEngine;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriber;
using ::mediapipe::tasks::audio::text_to_speech::AudioResult;
using ::mediapipe::tasks::audio::text_to_speech::SynthesisOptions;
using ::mediapipe::tasks::audio::text_to_speech::TextToSpeech;
using ::odml::audio::transcription::SpeechProcessor;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr float kMaxWaveformAbsDiff = 1e-5f;
constexpr double kMinWaveformSnrDb = 100.0;
constexpr size_t kHalfSecondSilenceSamples24kHz = 12000;
constexpr size_t kMinMultilingualSamples = 4000;
constexpr double kMinNonEuroDurationSec = 1.0;
constexpr double kMinNonEuroRmsEnergy = 0.02;

struct MultilingualTtsCase {
  absl::string_view lang_label;
  absl::string_view voice;
  absl::string_view locale;
  absl::string_view prompt;
  bool is_non_european = false;
};

struct MultilingualRoundTripCase {
  absl::string_view lang_label;
  absl::string_view voice;
  absl::string_view locale;
  absl::string_view prompt;
  absl::string_view expected_transcript;
  double max_allowed_wer;
};

std::vector<MultilingualTtsCase> GetMultilingualTtsCases() {
  return {
      {"1. en-US (American)", "am_adam", "en-US",
       "Hello world, speech synthesis runs fast on edge devices.", false},
      {"2. en-GB (British)", "bf_emma", "en-GB",
       "Good afternoon, the weather in London is rather lovely today.", false},
      {"3. es-ES (Spanish)", "ef_dora", "es-ES",
       "¡Buenos días! Esta es una prueba del sintetizador de voz en español.",
       false},
      {"4. fr-FR (French)", "ff_siwis", "fr-FR",
       "Bonjour le monde, voici un test de synthèse vocale en français.",
       false},
      {"5. hi-IN (Hindi)", "hf_alpha", "hi-IN",
       "नमस्ते भारत, यह हिंदी में आवाज़ संश्लेषण का एक परीक्षण है।", true},
      {"6. it-IT (Italian)", "if_sara", "it-IT",
       "Ciao mondo, questo è un test del sintetizzatore vocale in italiano.",
       false},
      {"7. pt-BR (Portuguese)", "pf_dora", "pt-BR",
       "Olá mundo, este é um teste do sintetizador de voz em português.",
       false},
      {"8. ja-JP (Japanese)", "jf_alpha", "ja-JP",
       "こんにちは世界、今日はとてもいい天気ですね。", true},
      {"9a. zh-CN (Simplified + Numbers)", "zf_xiaobei", "zh-CN",
       "你好世界，今天是2026年10月17日，天气非常好。", true},
      {"9b. zh-TW (Traditional zh-t2s)", "zm_yunjian", "zh-CN",
       "妳好世界，人工智慧語音合成技術非常自然。", true},
  };
}

std::vector<MultilingualRoundTripCase> GetEuropeanRoundTripCases() {
  return {
      {"en-US", "af_heart", "en-US",
       "Speech recognition and text to speech work together seamlessly.",
       "speech recognition and text to speech work together seamlessly", 0.20},
      {"en-GB", "bf_emma", "en-GB",
       "Good afternoon, the weather in London is rather lovely today.",
       "good afternoon the weather in london is rather lovely today", 0.25},
      {"es-ES", "ef_dora", "es-ES",
       "Hola amigos, buenos días a todos en la casa.",
       "hola amigos buenos dias a todos en la casa", 0.35},
      {"fr-FR", "ff_siwis", "fr-FR",
       "Bonjour tout le monde, comment allez vous aujourd'hui.",
       "bonjour tout le monde comment allez vous aujourd hui", 0.40},
      {"it-IT", "if_sara", "it-IT",
       "Ciao a tutti, oggi il tempo a Roma è molto bello.",
       "ciao a tutti oggi il tempo a roma e molto bello", 0.35},
      {"pt-BR", "pf_dora", "pt-BR",
       "Olá amigos, bom dia para todos vocês hoje.",
       "ola amigos bom dia para todos voces hoje", 0.40},
  };
}

void VerifyKokoroUnaryAndStreaming(OmniEngine& omni_engine,
                                   TextToSpeech& mp_tts) {
  for (absl::string_view prompt :
       {"Hello world, this is a test of the Kokoro speech synthesizer.",
        "What color is cheese? It is usually yellow or white."}) {
    absl::Time t_omni = absl::Now();
    ABSL_MP_ASSERT_OK_AND_ASSIGN(std::vector<float> omni_pcm,
                                 RunOmniKokoroSynthesize(omni_engine, prompt));
    double omni_ms = absl::ToDoubleMilliseconds(absl::Now() - t_omni);
    absl::Time t_mp = absl::Now();
    ABSL_MP_ASSERT_OK_AND_ASSIGN(AudioResult mp_res,
                                 mp_tts.Synthesize(std::string(prompt)));
    double mp_ms = absl::ToDoubleMilliseconds(absl::Now() - t_mp);
    WaveformComparison cmp = CompareWaveforms(omni_pcm, mp_res.samples);
    EXPECT_EQ(cmp.omni_samples, cmp.mp_samples);
    EXPECT_LE(cmp.max_abs_diff, kMaxWaveformAbsDiff);
    EXPECT_GE(cmp.snr_db, kMinWaveformSnrDb);
    std::cout << absl::StrFormat(
        "  [Cached Unary] Omni: %6.1f ms | MP: %6.1f ms | Diff: %+5.1f%% | "
        "SNR: %.1f dB | \"%s\"\n",
        omni_ms, mp_ms, ((mp_ms - omni_ms) / omni_ms) * 100.0, cmp.snr_db,
        prompt);
  }
  constexpr absl::string_view kStreamParagraph =
      "First sentence is synthesized immediately. Second sentence follows! "
      "Third sentence tests cadence. Fourth sentence completes the stream.";
  absl::Time t_omni_para = absl::Now();
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      std::vector<float> omni_para_pcm,
      RunOmniKokoroSynthesize(omni_engine, kStreamParagraph));
  double omni_para_ms = absl::ToDoubleMilliseconds(absl::Now() - t_omni_para);

  absl::Time t_mp_para = absl::Now();
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      AudioResult mp_para_res,
      mp_tts.Synthesize(std::string(kStreamParagraph)));
  double mp_para_ms = absl::ToDoubleMilliseconds(absl::Now() - t_mp_para);
  EXPECT_EQ(omni_para_pcm.size(), mp_para_res.samples.size());

  absl::Time t_start = absl::Now();
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      auto mp_stream, mp_tts.SynthesizeStream(std::string(kStreamParagraph)));
  double ttfa_ms = 0.0;
  int chunks_read = 0;
  while (true) {
    auto chunk_or = mp_stream->ReadNext();
    if (absl::IsOutOfRange(chunk_or.status())) break;
    ABSL_MP_ASSERT_OK(chunk_or.status());
    if (chunks_read++ == 0) {
      ttfa_ms = absl::ToDoubleMilliseconds(absl::Now() - t_start);
    }
  }
  double mp_stream_total_ms = absl::ToDoubleMilliseconds(absl::Now() - t_start);
  EXPECT_EQ(chunks_read, 4);
  EXPECT_LT(ttfa_ms, mp_stream_total_ms * 0.5);
  std::cout << absl::StrFormat(
      "  [4-Clause Paragraph (Equalized Cache)] Omni Sequential: %6.1f ms | "
      "MP Pipelined Unary: %6.1f ms (%+5.1f%%) | MP Stream: TTFA=%5.1f ms, "
      "Total=%6.1f ms (%+5.1f%%)\n",
      omni_para_ms, mp_para_ms,
      ((mp_para_ms - omni_para_ms) / omni_para_ms) * 100.0, ttfa_ms,
      mp_stream_total_ms,
      ((mp_stream_total_ms - omni_para_ms) / omni_para_ms) * 100.0);
}

void VerifyKokoroMultilingualAndVoiceSwitchParity(absl::string_view kokoro_path,
                                                  TextToSpeech& mp_tts) {
  for (const auto& tc : GetMultilingualTtsCases()) {
    absl::Time t_omni_sw = absl::Now();
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        auto omni_engine,
        CreateOmniKokoroEngine(kokoro_path, tc.voice, tc.locale));
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        std::vector<float> omni_pcm,
        RunOmniKokoroSynthesize(*omni_engine, tc.prompt));
    double omni_sw_ms = absl::ToDoubleMilliseconds(absl::Now() - t_omni_sw);
    absl::Time t_omni_cached = absl::Now();
    ABSL_MP_ASSERT_OK(
        RunOmniKokoroSynthesize(*omni_engine, tc.prompt).status());
    double omni_cached_ms =
        absl::ToDoubleMilliseconds(absl::Now() - t_omni_cached);

    SynthesisOptions opts{.voice_id = std::string(tc.voice),
                          .locale = std::string(tc.locale)};
    absl::Time t_mp_sw = absl::Now();
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AudioResult mp_sw, mp_tts.Synthesize(std::string(tc.prompt), opts));
    double mp_sw_ms = absl::ToDoubleMilliseconds(absl::Now() - t_mp_sw);
    absl::Time t_mp_cached = absl::Now();
    ABSL_MP_ASSERT_OK(mp_tts.Synthesize(std::string(tc.prompt), opts).status());
    double mp_cached_ms = absl::ToDoubleMilliseconds(absl::Now() - t_mp_cached);

    WaveformComparison cmp = CompareWaveforms(omni_pcm, mp_sw.samples);
    EXPECT_GT(mp_sw.samples.size(), kMinMultilingualSamples) << tc.lang_label;
    EXPECT_EQ(cmp.omni_samples, cmp.mp_samples) << tc.lang_label;
    EXPECT_LE(cmp.max_abs_diff, kMaxWaveformAbsDiff) << tc.lang_label;
    EXPECT_GE(cmp.snr_db, kMinWaveformSnrDb) << tc.lang_label;
    std::cout << absl::StrFormat(
        "  [%-32s | %-10s] Uncached Switch: Omni=%6.1f ms, MP=%5.1f ms | "
        "Cached: Omni=%5.1f ms, MP=%5.1f ms | SNR: %.1f dB\n",
        tc.lang_label, tc.voice, omni_sw_ms, mp_sw_ms, omni_cached_ms,
        mp_cached_ms, cmp.snr_db);
  }
}

void VerifyEnglishLexiconVsRuleFallback(TextToSpeech& tts_base,
                                        TextToSpeech& tts_no_espeak,
                                        AudioTranscriber& mp_asr) {
  const std::vector<absl::string_view> en_utterances = {
      "what color is cheese",
      "hello is there anybody in there",
      "it should not miss the beginning of the audio stream",
      "there is something in the water near the shore",
      "the quick brown fox jumps over the lazy dog",
      "artificial intelligence runs fast on edge devices",
      "speech recognition and text to speech work together seamlessly",
  };
  auto eval_wer = [&](TextToSpeech& tts,
                      absl::string_view expected) -> absl::StatusOr<double> {
    ABSL_ASSIGN_OR_RETURN(AudioResult audio,
                          tts.Synthesize(absl::StrCat(expected, ".")));
    audio.samples.insert(audio.samples.end(), kHalfSecondSilenceSamples24kHz,
                         0.0f);
    ABSL_ASSIGN_OR_RETURN(
        AsrRunMetrics rt,
        RunMediaPipeAsr(mp_asr, audio.samples, kSampleRate24kHz,
                        kHalfSecondSilenceSamples24kHz));
    return ComputeWordErrorRate(expected,
                                NormalizeTranscript(rt.final_transcript));
  };
  std::vector<double> base_wers, lexicon_wers;
  for (absl::string_view expected : en_utterances) {
    ABSL_MP_ASSERT_OK_AND_ASSIGN(double base_wer, eval_wer(tts_base, expected));
    ABSL_MP_ASSERT_OK_AND_ASSIGN(double lexicon_wer,
                                 eval_wer(tts_no_espeak, expected));
    base_wers.push_back(base_wer);
    lexicon_wers.push_back(lexicon_wer);
  }
  EXPECT_LE(Mean(lexicon_wers), Mean(base_wers));
}

void VerifyEuropeanMultilingualRoundTrip(absl::string_view kokoro_path,
                                         TextToSpeech& tts_no_espeak,
                                         SpeechProcessor& omni_asr,
                                         AudioTranscriber& mp_asr) {
  std::vector<double> multilingual_wers;
  for (const auto& tc : GetEuropeanRoundTripCases()) {
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        auto omni_tts,
        CreateOmniKokoroEngine(kokoro_path, tc.voice, tc.locale));
    ABSL_MP_ASSERT_OK_AND_ASSIGN(std::vector<float> omni_pcm,
                                 RunOmniKokoroSynthesize(*omni_tts, tc.prompt));
    SynthesisOptions opts{.voice_id = std::string(tc.voice),
                          .locale = std::string(tc.locale)};
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AudioResult synth_audio,
        tts_no_espeak.Synthesize(std::string(tc.prompt), opts));
    ASSERT_EQ(synth_audio.sample_rate_hz, kSampleRate24kHz);
    ASSERT_THAT(synth_audio.samples, Not(IsEmpty()));
    WaveformComparison tts_cmp =
        CompareWaveforms(omni_pcm, synth_audio.samples);
    EXPECT_EQ(tts_cmp.omni_samples, tts_cmp.mp_samples) << tc.lang_label;
    EXPECT_LE(tts_cmp.max_abs_diff, kMaxWaveformAbsDiff) << tc.lang_label;

    synth_audio.samples.insert(synth_audio.samples.end(),
                               kHalfSecondSilenceSamples24kHz, 0.0f);
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics omni_rt,
        RunOmniAsr(omni_asr, synth_audio.samples, kSampleRate24kHz,
                   kHalfSecondSilenceSamples24kHz));
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics mp_rt,
        RunMediaPipeAsr(mp_asr, synth_audio.samples, kSampleRate24kHz,
                        kHalfSecondSilenceSamples24kHz));
    std::string norm_omni = NormalizeTranscript(omni_rt.final_transcript);
    std::string norm_mp = NormalizeTranscript(mp_rt.final_transcript);
    EXPECT_EQ(norm_mp, norm_omni) << tc.lang_label;
    double wer = ComputeWordErrorRate(tc.expected_transcript, norm_mp);
    multilingual_wers.push_back(wer);
    EXPECT_LE(wer, tc.max_allowed_wer) << tc.lang_label << " Got=" << norm_mp;
  }
  std::cout << absl::StrFormat(
      "  >>> Mean 6-Language Closed-Loop WER: %.2f%%\n",
      Mean(multilingual_wers) * 100.0);
}

void VerifyNonEuropeanPhonemizerAndWaveformEval(TextToSpeech& tts_no_espeak) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::string qwen_asr_path,
                               ResolveStagedModel(kQwen3AsrModelName));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AudioTranscriber> qwen_asr,
      CreatePrewarmedMediaPipeAsr(qwen_asr_path, kSampleRate24kHz));
  for (const auto& tc : GetMultilingualTtsCases()) {
    if (!tc.is_non_european) continue;
    SynthesisOptions opts{.voice_id = std::string(tc.voice),
                          .locale = std::string(tc.locale)};
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AudioResult mp_res,
        tts_no_espeak.Synthesize(std::string(tc.prompt), opts));
    EXPECT_GT(static_cast<double>(mp_res.samples.size()) / kSampleRate24kHz,
              kMinNonEuroDurationSec)
        << tc.lang_label;
    EXPECT_GT(ComputeRmsEnergy(mp_res.samples), kMinNonEuroRmsEnergy)
        << tc.lang_label;
    mp_res.samples.insert(mp_res.samples.end(), kHalfSecondSilenceSamples24kHz,
                          0.0f);
    ABSL_MP_ASSERT_OK_AND_ASSIGN(
        AsrRunMetrics asr_m,
        RunMediaPipeAsr(*qwen_asr, mp_res.samples, kSampleRate24kHz, 0));
    EXPECT_FALSE(asr_m.final_transcript.empty()) << tc.lang_label;
    std::cout << absl::StrFormat("  [Asian TTS->Qwen3-ASR %-28s] \"%s\"\n",
                                 tc.lang_label, asr_m.final_transcript);
  }
  ABSL_MP_ASSERT_OK(qwen_asr->Close());
}

TEST(TtsOmniVsMediaPipeEvalTest,
     KokoroTtsOmniVsMediaPipeMultilingualAndStreamingParity) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::string kokoro_path,
                               ResolveStagedModel(kKokoroNoEspeakModelName));

  int threads_before_omni = GetCurrentProcessThreadCount();
  absl::Time t_omni_init = absl::Now();
  auto staged_wrapper_pool = std::make_unique<::litert::lm::ThreadPool>(
      "mediapipe_tts_cc_pool", kDefaultCpuThreads);
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      auto omni_engine, CreateOmniKokoroEngine(kokoro_path, kDefaultKokoroVoice,
                                               kDefaultKokoroLocale));
  double omni_init_ms = absl::ToDoubleMilliseconds(absl::Now() - t_omni_init);
  int omni_threads =
      std::max(0, GetCurrentProcessThreadCount() - threads_before_omni);

  int threads_before_mp = GetCurrentProcessThreadCount();
  absl::Time t_mp_init = absl::Now();
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextToSpeech> mp_tts,
                               CreateMediaPipeKokoroTts(kokoro_path));
  double mp_init_ms = absl::ToDoubleMilliseconds(absl::Now() - t_mp_init);
  int mp_threads =
      std::max(0, GetCurrentProcessThreadCount() - threads_before_mp);
  std::cout << absl::StrFormat(
      "\nInit Latency & Threads: Omni = %.1f ms (+%d) | MP = %.1f ms (+%d)\n",
      omni_init_ms, omni_threads, mp_init_ms, mp_threads);

  ABSL_MP_ASSERT_OK(RunOmniKokoroSynthesize(*omni_engine, "Warmup.").status());
  ABSL_MP_ASSERT_OK(mp_tts->Synthesize("Warmup.").status());
  VerifyKokoroUnaryAndStreaming(*omni_engine, *mp_tts);
  VerifyKokoroMultilingualAndVoiceSwitchParity(kokoro_path, *mp_tts);
  ABSL_MP_ASSERT_OK(mp_tts->Close());
}

TEST(TtsOmniVsMediaPipeEvalTest,
     MultilingualTtsToAsrRoundtripPipelineVerification) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::string base_path,
                               ResolveStagedModel(kKokoroBaseModelName));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::string kokoro_path,
                               ResolveStagedModel(kKokoroNoEspeakModelName));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::string tdt_path,
                               ResolveStagedModel(kParakeetTdtModelName));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SpeechProcessor> omni_asr,
      CreatePrewarmedOmniAsr(tdt_path, kSampleRate24kHz));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AudioTranscriber> mp_asr,
      CreatePrewarmedMediaPipeAsr(tdt_path, kSampleRate24kHz));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextToSpeech> tts_base,
                               CreateMediaPipeKokoroTts(base_path));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextToSpeech> tts_no_espeak,
                               CreateMediaPipeKokoroTts(kokoro_path));
  VerifyEnglishLexiconVsRuleFallback(*tts_base, *tts_no_espeak, *mp_asr);
  VerifyEuropeanMultilingualRoundTrip(kokoro_path, *tts_no_espeak, *omni_asr,
                                      *mp_asr);
  VerifyNonEuropeanPhonemizerAndWaveformEval(*tts_no_espeak);
  ABSL_MP_ASSERT_OK(mp_asr->Close());
  ABSL_MP_ASSERT_OK(tts_base->Close());
  ABSL_MP_ASSERT_OK(tts_no_espeak->Close());
}

TEST(TtsOmniVsMediaPipeEvalTest, Qwen3TtsCodecLmRvqMediaPipeEval) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(std::string qwen_tts_path,
                               ResolveStagedModel(kQwen3TtsModelName));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TextToSpeech> qwen_tts,
      CreateMediaPipeKokoroTts(qwen_tts_path, "qwen3_default", "English"));
  absl::Time t_start = absl::Now();
  ABSL_MP_ASSERT_OK_AND_ASSIGN(AudioResult res, qwen_tts->Synthesize("Hello."));
  double synth_ms = absl::ToDoubleMilliseconds(absl::Now() - t_start);
  EXPECT_EQ(res.sample_rate_hz, kSampleRate24kHz);
  EXPECT_THAT(res.samples, Not(IsEmpty()));
  std::cout << absl::StrFormat(
      "\n[Qwen3-TTS-0.6B INT4 (Apache-2.0 | kCodecLmRvq)] Samples=%zu "
      "(%.2f s @ %d Hz) | Latency=%.1f ms\n",
      res.samples.size(),
      static_cast<double>(res.samples.size()) / res.sample_rate_hz,
      res.sample_rate_hz, synth_ms);
  ABSL_MP_ASSERT_OK(qwen_tts->Close());
}

}  // namespace
}  // namespace mediapipe::tasks::audio::benchmarks
