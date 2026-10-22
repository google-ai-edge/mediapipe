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

#include "mediapipe/tasks/cc/audio/benchmarks/eval_benchmark_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/vector.h"
#include "mediapipe/tasks/cc/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/cc/audio/audio_transcriber/mediapipe_asr_processor.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/mediapipe_tts_session.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/phonon/common.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/pocket_tts/common.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/text_to_speech.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "odml/audio/transcription/omni/speech_processor_omni.h"  // from @odml
#include "odml/audio/transcription/proto/omni_options.pb.h"       // from @odml
#include "odml/audio/transcription/speech_processor.h"            // from @odml
#include "odml/audio/transcription/transcription_result.h"        // from @odml
#include "omni/base/io_types.h"                            // from @litert_lm
#include "omni/base/model_utils.h"                         // from @litert_lm
#include "omni/multi_staged_session.h"                     // from @litert_lm
#include "omni/omni_engine.h"                              // from @litert_lm
#include "omni/omni_session.h"                             // from @litert_lm
#include "omni/text2speech/stream_text_source.h"           // from @litert_lm
#include "runtime/proto/text2speech_metadata.pb.h"         // from @litert_lm
#include "runtime/proto/text2speech_model_type.pb.h"       // from @litert_lm
#include "runtime/proto/text2speech_session_config.pb.h"   // from @litert_lm
#include "schema/core/litertlm_header_schema_generated.h"  // from @litert_lm
#include "tflite/schema/schema_generated.h"

ABSL_FLAG(std::string, cache_dir,
          std::string(mediapipe::tasks::audio::benchmarks::kDefaultCacheDir),
          "Directory containing staged .litertlm models.");

namespace mediapipe::tasks::audio::benchmarks {
namespace {

using ::litert::omni::AudioOutput;
using ::litert::omni::MultiStagedSession;
using ::litert::omni::OmniEngine;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriber;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriberOptions;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriberResult;
using ::mediapipe::tasks::audio::text_to_speech::TextToSpeech;
using ::odml::audio::transcription::OmniSpeechProcessorOptions;
using ::odml::audio::transcription::SpeechProcessor;
using ::odml::audio::transcription::TranscriptionResult;

constexpr double kMinNoisePower = 1e-20;
constexpr double kMaxSnrDb = 200.0;
constexpr double kMsPerSecond = 1000.0;
constexpr absl::string_view kLatin1FoldTable =
    "aaaaaa.ceeeeiiii.nooooo..uuuuy.y";

bool FoldLatin1Diacritic(unsigned char c2, std::string& out) {
  if (c2 == 0x9F) {
    out.append("ss");
    return true;
  }
  const unsigned char lower = static_cast<unsigned char>(c2 | 0x20);
  if (lower == 0xA6) {
    out.append("ae");
    return true;
  }
  if (lower < 0xA0 || lower > 0xBF) return false;
  const char folded = kLatin1FoldTable[lower - 0xA0];
  if (folded == '.') return false;
  out.push_back(folded);
  return true;
}

const auto& ExtractSegments(const std::vector<TranscriptionResult>& res) {
  return res;
}
const auto& ExtractSegments(const AudioTranscriberResult& res) {
  return res.segments;
}

template <typename SegmentContainer>
void CollectSegments(const SegmentContainer& segments, AsrRunMetrics& metrics,
                     bool count_interims) {
  for (const auto& seg : segments) {
    if (seg.is_final) {
      AppendSegmentTranscript(metrics.final_transcript, seg.transcript);
    } else if (count_interims) {
      ++metrics.interim_updates;
    }
  }
}

template <typename Engine>
absl::StatusOr<AsrRunMetrics> RunAsrEngine(Engine& engine,
                                           absl::Span<const float> samples,
                                           int sample_rate_hz,
                                           size_t chunk_samples) {
  AsrRunMetrics metrics;
  std::vector<double> chunk_ms;
  const double dur_s = static_cast<double>(samples.size()) / sample_rate_hz;
  const bool is_unary = (chunk_samples == 0);
  const size_t step = is_unary ? samples.size() : chunk_samples;
  absl::Time t_start = absl::Now();
  for (size_t off = 0; off < samples.size(); off += step) {
    size_t n = std::min(step, samples.size() - off);
    absl::Time c0 = absl::Now();
    ABSL_ASSIGN_OR_RETURN(auto res,
                          engine.Transcribe(samples.subspan(off, n), is_unary));
    chunk_ms.push_back(absl::ToDoubleMilliseconds(absl::Now() - c0));
    CollectSegments(ExtractSegments(res), metrics, true);
  }
  if (!is_unary) {
    absl::Time f0 = absl::Now();
    ABSL_ASSIGN_OR_RETURN(auto final_res, engine.FinishStream());
    metrics.finish_stream_ms = absl::ToDoubleMilliseconds(absl::Now() - f0);
    CollectSegments(ExtractSegments(final_res), metrics, false);
  }
  metrics.total_latency_ms = absl::ToDoubleMilliseconds(absl::Now() - t_start);
  metrics.p50_chunk_ms = Percentile(chunk_ms, 0.50);
  metrics.p90_chunk_ms = Percentile(chunk_ms, 0.90);
  metrics.max_chunk_ms = Percentile(chunk_ms, 1.00);
  metrics.rtf = (metrics.total_latency_ms / kMsPerSecond) / dur_s;
  return metrics;
}

}  // namespace

std::string GetBenchmarkCacheDir() {
  std::string dir = absl::GetFlag(FLAGS_cache_dir);
  return dir.empty() ? std::string(kDefaultCacheDir) : dir;
}

absl::StatusOr<std::string> ResolveStagedModel(absl::string_view filename) {
  if (filename == kPocketTtsModelName) {
    return EnsurePocketTtsLitertLmModel();
  }
  if (filename == kPhononModelName) {
    return EnsurePhononLitertLmModel();
  }
  std::string cache_dir = GetBenchmarkCacheDir();
  std::string local = absl::StrCat(cache_dir, "/", filename);
  if (std::ifstream(local).good()) return local;
  std::string remote = absl::StrCat(
      "/", "cns", "/py-d/home/tensorflow-mobile-eng/litert-omni/models/",
      filename);
  return ::litert::omni::StageCnsFileIfNeeded(remote, cache_dir);
}

int GetCurrentProcessThreadCount() {
  std::ifstream status_file("/proc/self/status");
  for (std::string line; std::getline(status_file, line);) {
    if (!absl::StartsWith(line, "Threads:")) continue;
    int count = 0;
    for (char c : absl::string_view(line).substr(8)) {
      if (absl::ascii_isdigit(c)) count = count * 10 + (c - '0');
    }
    return count;
  }
  return -1;
}

std::string NormalizeTranscript(absl::string_view text) {
  std::string out;
  bool last_was_space = true;
  for (size_t i = 0; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (absl::ascii_isalnum(c)) {
      out.push_back(absl::ascii_tolower(c));
      last_was_space = false;
    } else if (c == 0xC3 && i + 1 < text.size()) {
      if (FoldLatin1Diacritic(static_cast<unsigned char>(text[++i]), out)) {
        last_was_space = false;
      }
    } else if (c == 0xC5 && i + 1 < text.size()) {
      const unsigned char c2 = static_cast<unsigned char>(text[++i]);
      if (c2 == 0x92 || c2 == 0x93) {
        out.append("oe");
        last_was_space = false;
      }
    } else if (c == 0xE2 && i + 2 < text.size()) {
      const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
      const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
      if (c1 == 0x80 &&
          (c2 == 0x99 || c2 == 0x98 || c2 == 0x93 || c2 == 0x94) &&
          !last_was_space) {
        out.push_back(' ');
        last_was_space = true;
      }
      i += 2;
    } else if ((absl::ascii_isspace(c) || c == '\'' || c == '-') &&
               !last_was_space) {
      out.push_back(' ');
      last_was_space = true;
    }
  }
  if (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

std::vector<std::string> SplitNormalizedWords(absl::string_view text) {
  std::string norm = NormalizeTranscript(text);
  if (norm.empty()) return {};
  static const auto* const kCanonicalTokenMap =
      new std::vector<std::pair<absl::string_view, absl::string_view>>{
          {"zero", "0"},
          {"one", "1"},
          {"two", "2"},
          {"three", "3"},
          {"four", "4"},
          {"five", "5"},
          {"six", "6"},
          {"seven", "7"},
          {"eight", "8"},
          {"nine", "9"},
          {"ten", "10"},
          {"eleven", "11"},
          {"twelve", "12"},
          {"thirteen", "13"},
          {"fourteen", "14"},
          {"fifteen", "15"},
          {"sixteen", "16"},
          {"seventeen", "17"},
          {"eighteen", "18"},
          {"nineteen", "19"},
          {"twenty", "20"},
          {"thirty", "30"},
          {"forty", "40"},
          {"fifty", "50"},
          {"sixty", "60"},
          {"seventy", "70"},
          {"eighty", "80"},
          {"ninety", "90"},
          {"hundred", "100"},
          {"un", "1"},
          {"deux", "2"},
          {"trois", "3"},
          {"quatre", "4"},
          {"cinq", "5"},
          {"sept", "7"},
          {"huit", "8"},
          {"neuf", "9"},
          {"dix", "10"},
          {"eins", "1"},
          {"zwei", "2"},
          {"drei", "3"},
          {"vier", "4"},
          {"funf", "5"},
          {"fuenf", "5"},
          {"sechs", "6"},
          {"sieben", "7"},
          {"acht", "8"},
          {"neun", "9"},
          {"zehn", "10"},
          {"zwanzig", "20"},
          {"uno", "1"},
          {"dos", "2"},
          {"tres", "3"},
          {"cuatro", "4"},
          {"cinco", "5"},
          {"seis", "6"},
          {"siete", "7"},
          {"ocho", "8"},
          {"nueve", "9"},
          {"diez", "10"},
          {"due", "2"},
          {"tre", "3"},
          {"quattro", "4"},
          {"cinque", "5"},
          {"sei", "6"},
          {"sette", "7"},
          {"otto", "8"},
          {"nove", "9"},
          {"dieci", "10"},
          {"dois", "2"},
          {"oito", "8"},
          {"dez", "10"},
          {"een", "1"},
          {"twee", "2"},
          {"drie", "3"},
          {"vijf", "5"},
          {"zes", "6"},
          {"zeven", "7"},
          {"negen", "9"},
          {"tien", "10"},
          {"colourful", "colorful"},
          {"colour", "color"},
          {"colours", "colors"},
          {"favouring", "favoring"},
          {"favour", "favor"},
          {"alsjeblieft", "alstublieft"},
      };
  std::vector<std::string> raw_words =
      absl::StrSplit(norm, ' ', absl::SkipEmpty());
  std::vector<std::string> out;
  out.reserve(raw_words.size());
  for (std::string& w : raw_words) {
    if (w.size() >= 2 && w.back() == 'h' &&
        std::all_of(w.begin(), w.end() - 1,
                    [](unsigned char c) { return absl::ascii_isdigit(c); })) {
      out.push_back(w.substr(0, w.size() - 1));
      out.push_back("heures");
      continue;
    }
    for (const auto& [from, to] : *kCanonicalTokenMap) {
      if (w == from) {
        w = std::string(to);
        break;
      }
    }
    out.push_back(std::move(w));
  }
  return out;
}

double ComputeWordErrorRate(absl::string_view reference,
                            absl::string_view hypothesis) {
  std::vector<std::string> ref = SplitNormalizedWords(reference);
  std::vector<std::string> hyp = SplitNormalizedWords(hypothesis);
  if (ref.empty()) return hyp.empty() ? 0.0 : 1.0;
  const size_t n = ref.size();
  const size_t m = hyp.size();
  std::vector<std::vector<size_t>> dp(n + 1, std::vector<size_t>(m + 1, 0));
  for (size_t i = 0; i <= n; ++i) dp[i][0] = i;
  for (size_t j = 0; j <= m; ++j) dp[0][j] = j;
  for (size_t i = 1; i <= n; ++i) {
    for (size_t j = 1; j <= m; ++j) {
      size_t cost = (ref[i - 1] == hyp[j - 1]) ? 0 : 1;
      dp[i][j] = std::min(
          {dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost});
    }
  }
  return static_cast<double>(dp[n][m]) / static_cast<double>(n);
}

double Percentile(std::vector<double> vals, double p) {
  if (vals.empty()) return 0.0;
  std::sort(vals.begin(), vals.end());
  const int64_t max_idx = static_cast<int64_t>(vals.size() - 1);
  const int64_t idx = std::clamp(
      static_cast<int64_t>(std::llround(p * max_idx)), int64_t{0}, max_idx);
  return vals[idx];
}

double Mean(absl::Span<const double> vals) {
  if (vals.empty()) return 0.0;
  return std::accumulate(vals.begin(), vals.end(), 0.0) /
         static_cast<double>(vals.size());
}

double ComputeRmsEnergy(absl::Span<const float> samples) {
  if (samples.empty()) return 0.0;
  double sum_sq = 0.0;
  for (float s : samples) sum_sq += static_cast<double>(s) * s;
  return std::sqrt(sum_sq / static_cast<double>(samples.size()));
}

WaveformComparison CompareWaveforms(absl::Span<const float> a,
                                    absl::Span<const float> b) {
  WaveformComparison cmp{a.size(), b.size(), 0.0f, 0.0};
  const size_t n = std::min(a.size(), b.size());
  double sig_pow = 0.0;
  double noise_pow = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float diff = std::abs(a[i] - b[i]);
    cmp.max_abs_diff = std::max(cmp.max_abs_diff, diff);
    sig_pow += static_cast<double>(a[i]) * a[i];
    noise_pow += static_cast<double>(diff) * diff;
  }
  cmp.snr_db = (noise_pow <= kMinNoisePower)
                   ? kMaxSnrDb
                   : 10.0 * std::log10(sig_pow / noise_pow);
  return cmp;
}

void AppendSegmentTranscript(std::string& dst, absl::string_view segment) {
  if (segment.empty()) return;
  if (!dst.empty() && dst.back() != ' ' && segment.front() != ' ') {
    dst.push_back(' ');
  }
  absl::StrAppend(&dst, segment);
}

absl::StatusOr<std::unique_ptr<SpeechProcessor>> CreatePrewarmedOmniAsr(
    absl::string_view model_path, int sample_rate_hz) {
  OmniSpeechProcessorOptions opts;
  opts.set_model_path(std::string(model_path));
  opts.set_sample_rate_hz(sample_rate_hz);
  opts.set_backend(OmniSpeechProcessorOptions::CPU);
  opts.set_overlap_ratio(kDefaultAsrOverlapRatio);
  opts.set_enable_interim_results(true);
  opts.mutable_advanced_options()->set_num_cpu_threads(kDefaultCpuThreads);
  opts.mutable_advanced_options()->set_cache_dir(GetBenchmarkCacheDir());
  ABSL_ASSIGN_OR_RETURN(
      auto proc,
      ::odml::audio::transcription::omni::CreateSpeechProcessorOmni(opts));
  ABSL_RETURN_IF_ERROR(proc->Prewarm());
  return proc;
}

absl::StatusOr<std::unique_ptr<AudioTranscriber>> CreatePrewarmedMediaPipeAsr(
    absl::string_view model_path, int sample_rate_hz) {
  absl::SetFlag(&FLAGS_mediapipe_asr_use_staged_scheduler, false);
  auto opts = std::make_unique<AudioTranscriberOptions>();
  opts->base_options.model_asset_path = std::string(model_path);
  opts->base_options.delegate = tasks::core::BaseOptions::Delegate::CPU;
  opts->base_options.delegate_options =
      tasks::core::BaseOptions::CpuOptions{.num_threads = kDefaultCpuThreads};
  opts->sample_rate_hz = sample_rate_hz;
  opts->model_architecture =
      ::mediapipe::tasks::audio::audio_transcriber::ModelArchitecture::kOmni;
  opts->enable_interim_results = true;
  opts->cache_dir = GetBenchmarkCacheDir();
  ABSL_ASSIGN_OR_RETURN(auto tr, AudioTranscriber::Create(std::move(opts)));
  ABSL_RETURN_IF_ERROR(tr->Prewarm());
  return tr;
}

absl::StatusOr<AsrRunMetrics> RunOmniAsr(SpeechProcessor& omni_proc,
                                         absl::Span<const float> audio_samples,
                                         int sample_rate_hz,
                                         size_t chunk_samples) {
  return RunAsrEngine(omni_proc, audio_samples, sample_rate_hz, chunk_samples);
}

absl::StatusOr<AsrRunMetrics> RunMediaPipeAsr(
    AudioTranscriber& mp_transcriber, absl::Span<const float> audio_samples,
    int sample_rate_hz, size_t chunk_samples) {
  return RunAsrEngine(mp_transcriber, audio_samples, sample_rate_hz,
                      chunk_samples);
}

absl::StatusOr<std::unique_ptr<OmniEngine>> CreateOmniKokoroEngine(
    absl::string_view model_path, absl::string_view voice,
    absl::string_view locale) {
  OmniEngine::Settings settings;
  settings.model_path = std::string(model_path);
  settings.cache_dir = GetBenchmarkCacheDir();
  settings.num_threads = kDefaultCpuThreads;
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  ::litert::lm::proto::Text2SpeechSessionConfig cfg;
  cfg.set_language(locale);
  cfg.set_voice(voice);
  ::litert::lm::proto::Text2SpeechMetadata meta;
  auto* kokoro_cfg = meta.mutable_tts_model_type()->mutable_kokoro();
  kokoro_cfg->set_voice_name(voice);
  kokoro_cfg->set_language(locale);
  return OmniEngine::Create("kokoro", settings, &cfg, &meta);
}

namespace {

float Bfloat16ToFloat32(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(float));
  return out;
}

uint16_t Float32ToBfloat16(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(float));
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
    return 0x7FC0u;
  }
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

bool FindSafetensorsOffsets(absl::string_view header_json,
                            absl::string_view tensor_key, uint64_t* start_out,
                            uint64_t* end_out, bool* is_bf16_out = nullptr) {
  std::string needle = absl::StrCat("\"", tensor_key, "\"");
  size_t key_pos = header_json.find(needle);
  if (key_pos == std::string::npos) return false;
  if (is_bf16_out != nullptr) {
    size_t dtype_pos = header_json.find("\"dtype\"", key_pos);
    size_t brace_end = header_json.find('}', key_pos);
    if (dtype_pos != std::string::npos && dtype_pos < brace_end) {
      *is_bf16_out = absl::StrContains(
          header_json.substr(dtype_pos, brace_end - dtype_pos), "BF16");
    }
  }
  size_t offsets_pos = header_json.find("\"data_offsets\"", key_pos);
  if (offsets_pos == std::string::npos) return false;
  size_t lb = header_json.find('[', offsets_pos);
  size_t comma = header_json.find(',', lb);
  size_t rb = header_json.find(']', comma);
  if (lb == std::string::npos || comma == std::string::npos ||
      rb == std::string::npos) {
    return false;
  }
  uint64_t start_off = 0;
  uint64_t end_off = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(
                            header_json.substr(lb + 1, comma - lb - 1)),
                        &start_off) ||
      !absl::SimpleAtoi(absl::StripAsciiWhitespace(
                            header_json.substr(comma + 1, rb - comma - 1)),
                        &end_off) ||
      end_off <= start_off) {
    return false;
  }
  *start_out = start_off;
  *end_out = end_off;
  return true;
}

std::vector<float> ReadSafetensorsSliceAsFloat(
    absl::string_view safetensors_path, absl::string_view tensor_key,
    size_t expected_floats, float scale = 1.0f) {
  std::vector<float> result(expected_floats, 0.0f);
  std::ifstream in(std::string(safetensors_path), std::ios::binary);
  if (!in.is_open()) return result;
  uint64_t header_len = 0;
  in.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));
  if (!in || header_len == 0 || header_len > (16u << 20)) {
    return result;
  }
  std::string header_json(header_len, '\0');
  in.read(header_json.data(), header_len);
  uint64_t start_off = 0;
  uint64_t end_off = 0;
  bool is_bf16 = true;
  if (!FindSafetensorsOffsets(header_json, tensor_key, &start_off, &end_off,
                              &is_bf16)) {
    return result;
  }
  in.seekg(static_cast<std::streamoff>(8 + header_len + start_off),
           std::ios::beg);
  if (is_bf16) {
    const size_t avail_elems = (end_off - start_off) / sizeof(uint16_t);
    const size_t to_read = std::min(expected_floats, avail_elems);
    std::vector<uint16_t> raw_bf16(to_read);
    in.read(reinterpret_cast<char*>(raw_bf16.data()),
            to_read * sizeof(uint16_t));
    for (size_t i = 0; i < to_read; ++i) {
      float v = Bfloat16ToFloat32(raw_bf16[i]);
      result[i] = std::isfinite(v) ? (v * scale) : 0.0f;
    }
  } else {
    const size_t avail_elems = (end_off - start_off) / sizeof(float);
    const size_t to_read = std::min(expected_floats, avail_elems);
    in.read(reinterpret_cast<char*>(result.data()), to_read * sizeof(float));
    for (size_t i = 0; i < to_read; ++i) {
      result[i] = std::isfinite(result[i]) ? (result[i] * scale) : 0.0f;
    }
  }
  return result;
}

struct SafetensorsEntry {
  std::string name;
  bool is_bf16 = false;
  uint64_t start_off = 0;
  uint64_t end_off = 0;
};

std::vector<SafetensorsEntry> ParseSafetensorsEntries(
    const std::string& header_json) {
  std::vector<SafetensorsEntry> entries;
  size_t pos = 0;
  while (pos < header_json.size()) {
    size_t q1 = header_json.find('"', pos);
    if (q1 == std::string::npos) break;
    size_t q2 = header_json.find('"', q1 + 1);
    if (q2 == std::string::npos) break;
    std::string key = header_json.substr(q1 + 1, q2 - q1 - 1);
    size_t colon = header_json.find(':', q2 + 1);
    if (colon == std::string::npos) break;
    size_t next_brace = header_json.find('{', colon);
    if (next_brace == std::string::npos) break;
    size_t close_brace = header_json.find('}', next_brace);
    if (close_brace == std::string::npos) break;
    pos = close_brace + 1;

    if (key == "__metadata__" ||
        absl::StartsWith(
            key,
            "flow_lm.condition_provider.conditioners.transcript_in_segment.") ||
        absl::StartsWith(
            key, "flow_lm.condition_provider.conditioners.speaker_wavs.") ||
        absl::StartsWith(key, "mimi.encoder.") ||
        absl::StartsWith(key, "mimi.encoder_transformer.") ||
        absl::StartsWith(key, "mimi.downsample.") ||
        key == "flow_lm.speaker_proj_weight" ||
        key == "flow_lm.bos_before_voice") {
      continue;
    }
    uint64_t s_off = 0, e_off = 0;
    bool is_bf16 = false;
    if (FindSafetensorsOffsets(header_json, key, &s_off, &e_off, &is_bf16)) {
      entries.push_back({key, is_bf16, s_off, e_off});
    }
  }
  return entries;
}

absl::StatusOr<std::vector<uint8_t>> BuildPocketTtsWeightsBinarySection(
    absl::string_view safetensors_path) {
  std::ifstream in(std::string(safetensors_path), std::ios::binary);
  if (!in.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("PocketTTS safetensors not found at: ", safetensors_path));
  }
  uint64_t header_len = 0;
  in.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));
  if (!in || header_len == 0 || header_len > (16u << 20)) {
    return absl::InvalidArgumentError("Invalid safetensors header length.");
  }
  std::string header_json(header_len, '\0');
  in.read(header_json.data(), header_len);
  std::vector<SafetensorsEntry> entries = ParseSafetensorsEntries(header_json);

  std::vector<uint8_t> out;
  out.reserve(120u << 20);
  auto append_bytes = [&](const void* ptr, size_t n) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(ptr);
    out.insert(out.end(), p, p + n);
  };
  append_bytes("PKTW", 4);
  uint32_t num_tensors = static_cast<uint32_t>(entries.size());
  append_bytes(&num_tensors, sizeof(uint32_t));

  std::vector<uint16_t> bf16_scratch;
  std::vector<float> scales_scratch;
  std::vector<int8_t> q_scratch;
  for (const auto& entry : entries) {
    const uint32_t name_len = static_cast<uint32_t>(entry.name.size());
    append_bytes(&name_len, sizeof(uint32_t));
    append_bytes(entry.name.data(), name_len);

    const uint64_t raw_bytes = entry.end_off - entry.start_off;
    in.seekg(static_cast<std::streamoff>(8 + header_len + entry.start_off),
             std::ios::beg);
    const uint32_t num_elems =
        static_cast<uint32_t>(raw_bytes / sizeof(uint16_t));
    bf16_scratch.resize(num_elems);
    in.read(reinterpret_cast<char*>(bf16_scratch.data()), raw_bytes);

    const bool quantize_int8 =
        entry.name == "flow_lm.conditioner.embed.weight" ||
        (absl::StartsWith(entry.name, "flow_lm.transformer.layers.") &&
         absl::EndsWith(entry.name, ".weight") &&
         !absl::StrContains(entry.name, "norm"));
    if (quantize_int8) {
      const uint32_t dtype = 2;
      const uint32_t cols =
          absl::EndsWith(entry.name, ".linear2.weight") ? 4096u : 1024u;
      const uint32_t rows = num_elems / cols;
      append_bytes(&dtype, sizeof(uint32_t));
      append_bytes(&num_elems, sizeof(uint32_t));
      append_bytes(&rows, sizeof(uint32_t));
      scales_scratch.resize(rows);
      q_scratch.resize(num_elems);
      for (uint32_t r = 0; r < rows; ++r) {
        const uint16_t* row_src =
            bf16_scratch.data() + static_cast<size_t>(r) * cols;
        float max_abs = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
          max_abs = std::max(max_abs, std::abs(Bfloat16ToFloat32(row_src[c])));
        }
        scales_scratch[r] = max_abs > 0.0f ? (max_abs / 127.0f) : 0.0f;
        const float inv = max_abs > 0.0f ? (127.0f / max_abs) : 0.0f;
        int8_t* row_dst = q_scratch.data() + static_cast<size_t>(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
          float v = Bfloat16ToFloat32(row_src[c]) * inv;
          row_dst[c] =
              static_cast<int8_t>(std::clamp(std::llround(v), -127LL, 127LL));
        }
      }
      append_bytes(scales_scratch.data(), rows * sizeof(float));
      append_bytes(q_scratch.data(), num_elems * sizeof(int8_t));
    } else {
      const uint32_t dtype = 1;
      append_bytes(&dtype, sizeof(uint32_t));
      append_bytes(&num_elems, sizeof(uint32_t));
      append_bytes(bf16_scratch.data(), raw_bytes);
    }
  }
  return out;
}

absl::StatusOr<std::vector<uint8_t>> BuildPhononWeightsBinarySection(
    absl::string_view safetensors_path) {
  std::ifstream in(std::string(safetensors_path), std::ios::binary);
  if (!in.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Phonon safetensors not found at: ", safetensors_path));
  }
  uint64_t header_len = 0;
  in.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));
  if (!in || header_len == 0 || header_len > (16u << 20)) {
    return absl::InvalidArgumentError("Invalid safetensors header length.");
  }
  std::string header_json(header_len, '\0');
  in.read(header_json.data(), header_len);
  std::vector<SafetensorsEntry> entries = ParseSafetensorsEntries(header_json);

  std::vector<uint8_t> out;
  out.reserve(80u << 20);
  auto append_bytes = [&](const void* ptr, size_t n) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(ptr);
    out.insert(out.end(), p, p + n);
  };
  append_bytes("PHNW", 4);
  uint32_t num_tensors = static_cast<uint32_t>(entries.size());
  append_bytes(&num_tensors, sizeof(uint32_t));

  std::vector<float> f32_scratch;
  std::vector<uint16_t> bf16_scratch;
  for (const auto& entry : entries) {
    const uint32_t name_len = static_cast<uint32_t>(entry.name.size());
    append_bytes(&name_len, sizeof(uint32_t));
    append_bytes(entry.name.data(), name_len);

    const uint64_t raw_bytes = entry.end_off - entry.start_off;
    in.seekg(static_cast<std::streamoff>(8 + header_len + entry.start_off),
             std::ios::beg);
    if (entry.is_bf16) {
      const uint32_t dtype = 1;
      const uint32_t num_elems =
          static_cast<uint32_t>(raw_bytes / sizeof(uint16_t));
      append_bytes(&dtype, sizeof(uint32_t));
      append_bytes(&num_elems, sizeof(uint32_t));
      bf16_scratch.resize(num_elems);
      in.read(reinterpret_cast<char*>(bf16_scratch.data()), raw_bytes);
      append_bytes(bf16_scratch.data(), raw_bytes);
    } else {
      const uint32_t num_elems =
          static_cast<uint32_t>(raw_bytes / sizeof(float));
      f32_scratch.resize(num_elems);
      in.read(reinterpret_cast<char*>(f32_scratch.data()), raw_bytes);
      const uint32_t dtype = (num_elems <= 2048) ? 0u : 1u;
      append_bytes(&dtype, sizeof(uint32_t));
      append_bytes(&num_elems, sizeof(uint32_t));
      if (dtype == 0) {
        append_bytes(f32_scratch.data(), raw_bytes);
      } else {
        bf16_scratch.resize(num_elems);
        for (uint32_t i = 0; i < num_elems; ++i) {
          bf16_scratch[i] = Float32ToBfloat16(f32_scratch[i]);
        }
        append_bytes(bf16_scratch.data(),
                     static_cast<size_t>(num_elems) * sizeof(uint16_t));
      }
    }
  }
  return out;
}

flatbuffers::Offset<tflite::Buffer> CreateFloatBuffer(
    flatbuffers::FlatBufferBuilder& b, absl::Span<const float> data) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(data.data());
  return tflite::CreateBuffer(
      b, b.CreateVector(bytes, data.size() * sizeof(float)));
}

std::vector<uint8_t> BuildPhononTextEmbeddingTflite(
    absl::string_view safetensors_path) {
  constexpr int kVocabSize = 4001;
  constexpr int kDim =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kHiddenDim;
  std::vector<float> emb_weights =
      ReadSafetensorsSliceAsFloat(safetensors_path,
                                  "flow_lm.condition_provider.conditioners."
                                  "transcript_in_segment.embed.weight",
                                  kVocabSize * kDim, 1.0f);

  flatbuffers::FlatBufferBuilder b;
  auto gather_op =
      tflite::CreateOperatorCode(b, tflite::BuiltinOperator_GATHER);
  auto opcodes_vec = b.CreateVector({gather_op});

  auto empty_buf = tflite::CreateBuffer(b);
  auto weights_buf = CreateFloatBuffer(b, emb_weights);
  auto buffers_vec = b.CreateVector({empty_buf, weights_buf});

  auto params_tensor = tflite::CreateTensor(
      b, b.CreateVector<int32_t>({kVocabSize, kDim}),
      tflite::TensorType_FLOAT32, /*buffer=*/1, b.CreateString("params"));
  auto indices_tensor = tflite::CreateTensor(
      b, b.CreateVector<int32_t>({1}), tflite::TensorType_INT32,
      /*buffer=*/0, b.CreateString("indices"));
  auto output_tensor = tflite::CreateTensor(
      b, b.CreateVector<int32_t>({1, kDim}), tflite::TensorType_FLOAT32,
      /*buffer=*/0, b.CreateString("output"));
  auto tensors_vec =
      b.CreateVector({params_tensor, indices_tensor, output_tensor});

  auto gather_opts =
      tflite::CreateGatherOptions(b, /*axis=*/0, /*batch_dims=*/0);
  auto op = tflite::CreateOperator(
      b, /*opcode_index=*/0, b.CreateVector<int32_t>({0, 1}),
      b.CreateVector<int32_t>({2}), tflite::BuiltinOptions_GatherOptions,
      gather_opts.Union());
  auto ops_vec = b.CreateVector({op});

  auto subgraph = tflite::CreateSubGraph(
      b, tensors_vec, b.CreateVector<int32_t>({1}),
      b.CreateVector<int32_t>({2}), ops_vec, b.CreateString("main"));
  auto subgraphs_vec = b.CreateVector({subgraph});

  auto in_map = tflite::CreateTensorMap(b, b.CreateString("indices"), 1);
  auto out_map = tflite::CreateTensorMap(b, b.CreateString("output"), 2);
  auto sig_def = tflite::CreateSignatureDef(
      b, b.CreateVector({in_map}), b.CreateVector({out_map}),
      b.CreateString("serving_default"), /*subgraph_index=*/0);

  auto model = tflite::CreateModel(
      b, /*version=*/3, opcodes_vec, subgraphs_vec,
      b.CreateString("phonon_text_embedding"), buffers_vec,
      /*metadata_buffer=*/0, /*metadata=*/0, b.CreateVector({sig_def}));
  tflite::FinishModelBuffer(b, model);
  return std::vector<uint8_t>(b.GetBufferPointer(),
                              b.GetBufferPointer() + b.GetSize());
}

std::vector<uint8_t> FinishSingleSubgraphTflite(
    flatbuffers::FlatBufferBuilder& b,
    flatbuffers::Offset<tflite::OperatorCode> opcode,
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers,
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors,
    std::vector<int32_t> inputs, std::vector<int32_t> outputs,
    flatbuffers::Offset<tflite::Operator> op,
    std::vector<flatbuffers::Offset<tflite::TensorMap>> in_maps,
    std::vector<flatbuffers::Offset<tflite::TensorMap>> out_maps,
    absl::string_view description) {
  auto subgraph = tflite::CreateSubGraph(
      b, b.CreateVector(tensors), b.CreateVector(inputs),
      b.CreateVector(outputs), b.CreateVector({op}), b.CreateString("main"));
  auto sig_def = tflite::CreateSignatureDef(
      b, b.CreateVector(in_maps), b.CreateVector(out_maps),
      b.CreateString("serving_default"), 0);
  auto model = tflite::CreateModel(
      b, 3, b.CreateVector({opcode}), b.CreateVector({subgraph}),
      b.CreateString(std::string(description)), b.CreateVector(buffers), 0, 0,
      b.CreateVector({sig_def}));
  tflite::FinishModelBuffer(b, model);
  return std::vector<uint8_t>(b.GetBufferPointer(),
                              b.GetBufferPointer() + b.GetSize());
}

std::vector<uint8_t> BuildPhononBackboneTflite() {
  constexpr int kDim =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kHiddenDim;
  flatbuffers::FlatBufferBuilder b;
  std::vector<float> zeros(kDim, 0.0f);
  auto add_opts =
      tflite::CreateAddOptions(b, tflite::ActivationFunctionType_NONE);
  return FinishSingleSubgraphTflite(
      b, tflite::CreateOperatorCode(b, tflite::BuiltinOperator_ADD),
      {tflite::CreateBuffer(b), CreateFloatBuffer(b, zeros)},
      {tflite::CreateTensor(b, b.CreateVector<int32_t>({1, kDim}),
                            tflite::TensorType_FLOAT32, 0,
                            b.CreateString("input_embeds")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({1, kDim}),
                            tflite::TensorType_FLOAT32, 1,
                            b.CreateString("zeros")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({1, kDim}),
                            tflite::TensorType_FLOAT32, 0,
                            b.CreateString("conditioning"))},
      {0}, {2},
      tflite::CreateOperator(
          b, 0, b.CreateVector<int32_t>({0, 1}), b.CreateVector<int32_t>({2}),
          tflite::BuiltinOptions_AddOptions, add_opts.Union()),
      {tflite::CreateTensorMap(b, b.CreateString("input_embeds"), 0)},
      {tflite::CreateTensorMap(b, b.CreateString("conditioning"), 2)},
      "phonon_backbone");
}

std::vector<uint8_t> BuildPhononFlowHeadTflite() {
  constexpr int kDim =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kHiddenDim;
  constexpr int kLatent =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kLatentDim;
  flatbuffers::FlatBufferBuilder b;
  std::vector<float> zeros(kLatent, 0.0f);
  auto add_opts =
      tflite::CreateAddOptions(b, tflite::ActivationFunctionType_NONE);
  return FinishSingleSubgraphTflite(
      b, tflite::CreateOperatorCode(b, tflite::BuiltinOperator_ADD),
      {tflite::CreateBuffer(b), CreateFloatBuffer(b, zeros)},
      {tflite::CreateTensor(b, b.CreateVector<int32_t>({1, kLatent}),
                            tflite::TensorType_FLOAT32, 0,
                            b.CreateString("a_latent_z")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({1, 1}),
                            tflite::TensorType_FLOAT32, 0,
                            b.CreateString("b_timestep_t")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({1, kDim}),
                            tflite::TensorType_FLOAT32, 0,
                            b.CreateString("c_conditioning")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({1, kLatent}),
                            tflite::TensorType_FLOAT32, 1,
                            b.CreateString("zeros")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({1, kLatent}),
                            tflite::TensorType_FLOAT32, 0,
                            b.CreateString("velocity"))},
      {0, 1, 2}, {4},
      tflite::CreateOperator(
          b, 0, b.CreateVector<int32_t>({0, 3}), b.CreateVector<int32_t>({4}),
          tflite::BuiltinOptions_AddOptions, add_opts.Union()),
      {tflite::CreateTensorMap(b, b.CreateString("a_latent_z"), 0),
       tflite::CreateTensorMap(b, b.CreateString("b_timestep_t"), 1),
       tflite::CreateTensorMap(b, b.CreateString("c_conditioning"), 2)},
      {tflite::CreateTensorMap(b, b.CreateString("velocity"), 4)},
      "phonon_flow_head");
}

std::vector<uint8_t> BuildPhononCodecTflite() {
  constexpr int kWindowFrames = 32;
  constexpr int kLatent =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kLatentDim;
  constexpr int kUpsample =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kAudioUpsampleFactor;
  flatbuffers::FlatBufferBuilder b;
  std::vector<float> w(kUpsample * kLatent, 0.0f);
  std::vector<float> bias(kUpsample, 0.0f);
  auto fc_none = tflite::CreateFullyConnectedOptions(
      b, tflite::ActivationFunctionType_NONE);
  return FinishSingleSubgraphTflite(
      b, tflite::CreateOperatorCode(b, tflite::BuiltinOperator_FULLY_CONNECTED),
      {tflite::CreateBuffer(b), CreateFloatBuffer(b, w),
       CreateFloatBuffer(b, bias)},
      {tflite::CreateTensor(
           b, b.CreateVector<int32_t>({kWindowFrames, kLatent}),
           tflite::TensorType_FLOAT32, 0, b.CreateString("latents")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({kUpsample, kLatent}),
                            tflite::TensorType_FLOAT32, 1, b.CreateString("w")),
       tflite::CreateTensor(b, b.CreateVector<int32_t>({kUpsample}),
                            tflite::TensorType_FLOAT32, 2, b.CreateString("b")),
       tflite::CreateTensor(
           b, b.CreateVector<int32_t>({kWindowFrames, kUpsample}),
           tflite::TensorType_FLOAT32, 0, b.CreateString("waveform"))},
      {0}, {3},
      tflite::CreateOperator(b, 0, b.CreateVector<int32_t>({0, 1, 2}),
                             b.CreateVector<int32_t>({3}),
                             tflite::BuiltinOptions_FullyConnectedOptions,
                             fc_none.Union()),
      {tflite::CreateTensorMap(b, b.CreateString("latents"), 0)},
      {tflite::CreateTensorMap(b, b.CreateString("waveform"), 3)},
      "phonon_mimi_codec");
}

struct PendingLitertLmSection {
  ::litert::lm::schema::AnySectionDataType data_type;
  std::string item_key;
  std::string item_val;
  std::vector<uint8_t> payload;
};

void AppendPocketTtsVoiceSections(
    absl::string_view hf_dir, std::vector<PendingLitertLmSection>& pending) {
  constexpr int kLayers = 6;
  constexpr int kDim = 1024;
  for (absl::string_view voice :
       {"alba", "marius", "javert", "jean", "fantine", "cosette", "eponine",
        "azelma", "estelle", "juergen", "lola", "rafael", "giovanni", "daan"}) {
    std::string v_path =
        absl::StrCat(hf_dir, "/embeddings/", voice, ".safetensors");
    if (!std::ifstream(v_path).good()) {
      v_path = absl::StrCat(hf_dir, "/embeddings_v2/", voice, ".safetensors");
    }
    std::ifstream in(v_path, std::ios::binary);
    if (!in.is_open()) continue;
    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));
    if (!in || header_len == 0 || header_len > (1u << 20)) continue;
    std::string header_json(header_len, '\0');
    in.read(header_json.data(), header_len);

    uint64_t s0 = 0, e0 = 0;
    if (!FindSafetensorsOffsets(
            header_json, "transformer.layers.0.self_attn/cache", &s0, &e0)) {
      continue;
    }
    const uint32_t frames =
        static_cast<uint32_t>((e0 - s0) / (2 * kDim * sizeof(float)));
    if (frames == 0) continue;

    std::vector<uint8_t> v_bytes(8 + static_cast<size_t>(kLayers) * 2 * frames *
                                         kDim * sizeof(uint16_t));
    std::memcpy(v_bytes.data(), "PKTV", 4);
    std::memcpy(v_bytes.data() + 4, &frames, sizeof(uint32_t));
    uint16_t* dst = reinterpret_cast<uint16_t*>(v_bytes.data() + 8);
    for (int l = 0; l < kLayers; ++l) {
      std::string key =
          absl::StrCat("transformer.layers.", l, ".self_attn/cache");
      std::vector<float> layer_kv =
          ReadSafetensorsSliceAsFloat(v_path, key, 2 * frames * kDim, 1.0f);
      for (size_t i = 0; i < layer_kv.size(); ++i) {
        dst[static_cast<size_t>(l) * 2 * frames * kDim + i] =
            Float32ToBfloat16(layer_kv[i]);
      }
    }
    pending.push_back(
        {::litert::lm::schema::AnySectionDataType_GenericBinaryData, "name",
         std::string(voice), std::move(v_bytes)});
  }
}

void AppendPhononVoiceSections(absl::string_view hf_dir,
                               absl::string_view safetensors_path,
                               std::vector<PendingLitertLmSection>& pending) {
  constexpr int kDim =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kHiddenDim;
  constexpr int kVoiceFrames =
      ::mediapipe::tasks::audio::text_to_speech::phonon::kDefaultVoiceFrames;
  std::vector<float> spk_proj_w = ReadSafetensorsSliceAsFloat(
      safetensors_path,
      "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
      kDim * kDim, 1.0f);

  for (absl::string_view voice :
       {"freya", "claire", "lukas", "sofia", "lucas"}) {
    std::string v_path =
        absl::StrCat(hf_dir, "/voices/", voice, ".safetensors");
    std::vector<float> speaker_wavs = ReadSafetensorsSliceAsFloat(
        v_path, "speaker_wavs", kDim * kVoiceFrames, 1.0f);
    std::vector<float> v_floats(kVoiceFrames * kDim, 0.0f);
    for (int t = 0; t < kVoiceFrames; ++t) {
      for (int o = 0; o < kDim; ++o) {
        float sum = 0.0f;
        const float* proj_row = spk_proj_w.data() + o * kDim;
        for (int c = 0; c < kDim; ++c) {
          sum += speaker_wavs[c * kVoiceFrames + t] * proj_row[c];
        }
        v_floats[t * kDim + o] = sum;
      }
    }
    const auto* v_ptr = reinterpret_cast<const uint8_t*>(v_floats.data());
    std::vector<uint8_t> v_bytes(v_ptr,
                                 v_ptr + v_floats.size() * sizeof(float));
    pending.push_back(
        {::litert::lm::schema::AnySectionDataType_GenericBinaryData, "name",
         std::string(voice), std::move(v_bytes)});
  }
}

absl::Status WriteLitertLmContainerFile(
    absl::string_view output_path,
    const std::vector<PendingLitertLmSection>& pending) {
  constexpr uint64_t kAlign = 16384;
  auto align_up = [](uint64_t off) -> uint64_t {
    return ((off + kAlign - 1) / kAlign) * kAlign;
  };

  uint64_t cur_offset = kAlign;
  std::vector<std::pair<uint64_t, uint64_t>> offsets;
  for (const auto& sec : pending) {
    cur_offset = align_up(cur_offset);
    offsets.push_back({cur_offset, cur_offset + sec.payload.size()});
    cur_offset += sec.payload.size();
  }

  flatbuffers::FlatBufferBuilder fb(4096);
  auto author_val = ::litert::lm::schema::CreateStringValue(
      fb, fb.CreateString("MediaPipe Audio Team"));
  auto author_kv = ::litert::lm::schema::CreateKeyValuePair(
      fb, fb.CreateString("author"), ::litert::lm::schema::VData_StringValue,
      author_val.Union());
  auto sys_meta = ::litert::lm::schema::CreateSystemMetadata(
      fb, fb.CreateVector({author_kv}));

  std::vector<flatbuffers::Offset<::litert::lm::schema::SectionObject>>
      sec_objs;
  for (size_t i = 0; i < pending.size(); ++i) {
    flatbuffers::Offset<flatbuffers::Vector<
        flatbuffers::Offset<::litert::lm::schema::KeyValuePair>>>
        items_vec = 0;
    if (!pending[i].item_key.empty()) {
      auto sval = ::litert::lm::schema::CreateStringValue(
          fb, fb.CreateString(pending[i].item_val));
      auto kv = ::litert::lm::schema::CreateKeyValuePair(
          fb, fb.CreateString(pending[i].item_key),
          ::litert::lm::schema::VData_StringValue, sval.Union());
      items_vec = fb.CreateVector({kv});
    }
    sec_objs.push_back(::litert::lm::schema::CreateSectionObject(
        fb, items_vec, offsets[i].first, offsets[i].second,
        pending[i].data_type));
  }
  auto sec_meta = ::litert::lm::schema::CreateSectionMetadata(
      fb, fb.CreateVector(sec_objs));
  fb.Finish(
      ::litert::lm::schema::CreateLiteRTLMMetaData(fb, sys_meta, sec_meta));

  std::ofstream out(std::string(output_path),
                    std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return absl::PermissionDeniedError(
        absl::StrCat("Failed to open output file: ", output_path));
  }
  out.write("LITERTLM", 8);
  uint32_t major = 1, minor = 0, patch = 0, pad = 0;
  out.write(reinterpret_cast<const char*>(&major), 4);
  out.write(reinterpret_cast<const char*>(&minor), 4);
  out.write(reinterpret_cast<const char*>(&patch), 4);
  out.write(reinterpret_cast<const char*>(&pad), 4);
  uint64_t header_end_offset = 32 + fb.GetSize();
  out.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  out.write(reinterpret_cast<const char*>(fb.GetBufferPointer()), fb.GetSize());

  for (size_t i = 0; i < pending.size(); ++i) {
    uint64_t pos = static_cast<uint64_t>(out.tellp());
    if (offsets[i].first > pos) {
      std::string zeros(offsets[i].first - pos, '\0');
      out.write(zeros.data(), zeros.size());
    }
    out.write(reinterpret_cast<const char*>(pending[i].payload.data()),
              pending[i].payload.size());
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> EnsurePocketTtsLitertLmModel() {
  std::string cache_dir = GetBenchmarkCacheDir();
  std::string output_path = absl::StrCat(cache_dir, "/", kPocketTtsModelName);
  std::string marker_path = absl::StrCat(output_path, ".v3");
  if (std::ifstream(output_path).good() && std::ifstream(marker_path).good()) {
    return output_path;
  }

  const std::string hf_dir = absl::StrCat(cache_dir, "/pocket_tts_hf");
  const std::string safetensors_path =
      absl::StrCat(hf_dir, "/model.safetensors");
  const std::string sp_path = absl::StrCat(hf_dir, "/tokenizer.model");

  std::ifstream sp_file(sp_path, std::ios::binary);
  if (!sp_file.is_open()) {
    return absl::NotFoundError(absl::StrCat(
        "PocketTTS SentencePiece tokenizer not found at: ", sp_path));
  }
  std::vector<uint8_t> sp_bytes((std::istreambuf_iterator<char>(sp_file)),
                                std::istreambuf_iterator<char>());

  ::litert::lm::proto::Text2SpeechMetadata tts_meta;
  std::string tts_meta_str = tts_meta.SerializeAsString();
  std::vector<uint8_t> tts_meta_bytes(tts_meta_str.begin(), tts_meta_str.end());
  ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> pocket_weights_bytes,
                        BuildPocketTtsWeightsBinarySection(safetensors_path));

  std::vector<PendingLitertLmSection> pending = {
      {::litert::lm::schema::AnySectionDataType_Text2SpeechMetadataProto, "",
       "", std::move(tts_meta_bytes)},
      {::litert::lm::schema::AnySectionDataType_SP_Tokenizer, "", "",
       std::move(sp_bytes)},
      {::litert::lm::schema::AnySectionDataType_GenericBinaryData, "name",
       std::string(::mediapipe::tasks::audio::text_to_speech::pocket_tts::
                       kPocketTtsWeightsSectionName),
       std::move(pocket_weights_bytes)},
  };
  AppendPocketTtsVoiceSections(hf_dir, pending);
  ABSL_RETURN_IF_ERROR(WriteLitertLmContainerFile(output_path, pending));

  std::ofstream marker(marker_path, std::ios::binary | std::ios::trunc);
  marker << "v3\n";
  return output_path;
}

absl::StatusOr<std::string> EnsurePhononLitertLmModel() {
  std::string cache_dir = GetBenchmarkCacheDir();
  std::string output_path = absl::StrCat(cache_dir, "/", kPhononModelName);
  std::string marker_path = absl::StrCat(output_path, ".v3");
  if (std::ifstream(output_path).good() && std::ifstream(marker_path).good()) {
    return output_path;
  }

  const std::string hf_dir = absl::StrCat(cache_dir, "/phonon_hf");
  const std::string safetensors_path =
      absl::StrCat(hf_dir, "/model.safetensors");
  const std::string sp_path = absl::StrCat(hf_dir, "/tokenizer.model");

  std::ifstream sp_file(sp_path, std::ios::binary);
  if (!sp_file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Phonon SentencePiece tokenizer not found at: ", sp_path));
  }
  std::vector<uint8_t> sp_bytes((std::istreambuf_iterator<char>(sp_file)),
                                std::istreambuf_iterator<char>());

  ::litert::lm::proto::Text2SpeechMetadata tts_meta;
  std::string tts_meta_str = tts_meta.SerializeAsString();
  std::vector<uint8_t> tts_meta_bytes(tts_meta_str.begin(), tts_meta_str.end());
  ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> phonon_weights_bytes,
                        BuildPhononWeightsBinarySection(safetensors_path));

  std::vector<PendingLitertLmSection> pending = {
      {::litert::lm::schema::AnySectionDataType_Text2SpeechMetadataProto, "",
       "", std::move(tts_meta_bytes)},
      {::litert::lm::schema::AnySectionDataType_SP_Tokenizer, "", "",
       std::move(sp_bytes)},
      {::litert::lm::schema::AnySectionDataType_TFLiteModel, "model_type",
       "tf_lite_text_embedding",
       BuildPhononTextEmbeddingTflite(safetensors_path)},
      {::litert::lm::schema::AnySectionDataType_TFLiteModel, "model_type",
       "tf_lite_acoustic", BuildPhononBackboneTflite()},
      {::litert::lm::schema::AnySectionDataType_TFLiteModel, "model_type",
       "tf_lite_mtp_drafter", BuildPhononFlowHeadTflite()},
      {::litert::lm::schema::AnySectionDataType_TFLiteModel, "model_type",
       "tf_lite_vocoder", BuildPhononCodecTflite()},
      {::litert::lm::schema::AnySectionDataType_GenericBinaryData, "name",
       std::string(::mediapipe::tasks::audio::text_to_speech::phonon::
                       kPhononWeightsSectionName),
       std::move(phonon_weights_bytes)},
  };
  AppendPhononVoiceSections(hf_dir, safetensors_path, pending);
  ABSL_RETURN_IF_ERROR(WriteLitertLmContainerFile(output_path, pending));

  std::ofstream marker(marker_path, std::ios::binary | std::ios::trunc);
  marker << "v3\n";
  return output_path;
}

absl::StatusOr<std::unique_ptr<TextToSpeech>> CreateMediaPipeKokoroTts(
    absl::string_view model_path, absl::string_view default_voice,
    absl::string_view default_locale) {
  absl::SetFlag(&FLAGS_mediapipe_tts_use_staged_scheduler, false);
  ::mediapipe::tasks::audio::text_to_speech::TextToSpeechOptions opts;
  opts.base_options.model_asset_path = std::string(model_path);
  opts.base_options.delegate = tasks::core::BaseOptions::Delegate::CPU;
  opts.base_options.delegate_options =
      tasks::core::BaseOptions::CpuOptions{.num_threads = kDefaultCpuThreads};
  opts.cache_directory = GetBenchmarkCacheDir();
  opts.default_voice_id = std::string(default_voice);
  opts.default_locale = std::string(default_locale);
  return ::mediapipe::tasks::audio::text_to_speech::CreateTextToSpeech(opts);
}

absl::StatusOr<std::vector<float>> RunOmniKokoroSynthesize(
    OmniEngine& omni_engine, absl::string_view text) {
  ABSL_ASSIGN_OR_RETURN(
      auto session, omni_engine.CreateSession(
                        std::make_unique<::litert::omni::PushInputSource>()));
  std::unique_ptr<MultiStagedSession> owned(
      static_cast<MultiStagedSession*>(session.release()));
  auto* src = static_cast<::litert::omni::tts::StreamTextSource*>(
      owned->stages()[0].get());
  ABSL_RETURN_IF_ERROR(src->PushText(text));
  src->Finish();

  std::vector<float> samples;
  auto append_if_audio = [&](const MultiStagedSession::Output& out) {
    if (const auto* audio = std::get_if<AudioOutput>(&out)) {
      samples.insert(samples.end(), audio->pcm_samples.begin(),
                     audio->pcm_samples.end());
    }
  };
  while (true) {
    auto out_or = owned->ProcessNext();
    if (!out_or.ok()) {
      if (absl::IsOutOfRange(out_or.status()) ||
          absl::IsNotFound(out_or.status())) {
        break;
      }
      return out_or.status();
    }
    append_if_audio(*out_or);
    if (std::holds_alternative<::litert::omni::EndOfOutput>(*out_or)) break;
  }
  if (auto flush_or = owned->Flush(); flush_or.ok()) {
    append_if_audio(*flush_or);
  }
  return samples;
}

}  // namespace mediapipe::tasks::audio::benchmarks
