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
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/audio/benchmarks/eval_benchmark_utils.h"
#include "omni/base/async_stage_scheduler.h"  // from @litert_lm
#include "omni/base/io_types.h"               // from @litert_lm
#include "omni/base/stage.h"                  // from @litert_lm
#include "omni/multi_staged_session.h"        // from @litert_lm
#include "runtime/framework/threadpool.h"     // from @litert_lm
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe::tasks::audio::benchmarks {
namespace {

using ::litert::omni::MultiStagedSession;

constexpr int kNumPackets = 10000;
constexpr int kPipelineChunks = 5;
constexpr absl::Duration kSimulatedStage1Work = absl::Milliseconds(12);
constexpr absl::Duration kSimulatedStage2Work = absl::Milliseconds(18);
constexpr double kUsPerMillisecond = 1000.0;
constexpr absl::Duration kSchedulerStopTimeout = absl::Seconds(10);

class MicrobenchSourceStage
    : public ::litert::omni::SingleThreadedStageWithDeque<int64_t> {
 public:
  explicit MicrobenchSourceStage(int total_items) : total_items_(total_items) {}

 protected:
  bool NeedScheduleInternal() const override { return emitted_ < total_items_; }
  absl::Status ScheduleInternal() override {
    absl::Cleanup c = [this] { SetState(State::kIdle); };
    if (emitted_ >= total_items_) {
      return absl::OutOfRangeError("Done");
    }
    PushOutput(++emitted_);
    return absl::OkStatus();
  }

 private:
  int total_items_ = 0;
  int emitted_ = 0;
};

class MicrobenchTransformStage
    : public ::litert::omni::SingleThreadedStageWithDeque<int64_t> {
 public:
  explicit MicrobenchTransformStage(
      ::litert::omni::Stage<int64_t>& upstream,
      absl::Duration simulated_work = absl::ZeroDuration())
      : upstream_(upstream), simulated_work_(simulated_work) {}

 protected:
  bool NeedScheduleInternal() const override { return upstream_.HasOutput(); }
  absl::Status ScheduleInternal() override {
    absl::Cleanup c = [this] { SetState(State::kIdle); };
    auto val = upstream_.GetOutput();
    if (absl::IsNotFound(val.status())) {
      return absl::OkStatus();
    } else if (!val.ok()) {
      return val.status();
    }
    if (simulated_work_ > absl::ZeroDuration()) {
      absl::SleepFor(simulated_work_);
    }
    PushOutput((*val) * 2 + 1);
    return absl::OkStatus();
  }

 private:
  ::litert::omni::Stage<int64_t>& upstream_;
  absl::Duration simulated_work_;
};

class MicrobenchSinkStage : public ::litert::omni::SingleThreadedStageWithDeque<
                                MultiStagedSession::Output> {
 public:
  explicit MicrobenchSinkStage(
      ::litert::omni::Stage<int64_t>& upstream,
      absl::Duration simulated_work = absl::ZeroDuration())
      : upstream_(upstream), simulated_work_(simulated_work) {}

 protected:
  bool NeedScheduleInternal() const override { return upstream_.HasOutput(); }
  absl::Status ScheduleInternal() override {
    absl::Cleanup c = [this] { SetState(State::kIdle); };
    auto val = upstream_.GetOutput();
    if (absl::IsNotFound(val.status())) {
      return absl::OkStatus();
    } else if (!val.ok()) {
      return val.status();
    }
    if (simulated_work_ > absl::ZeroDuration()) {
      absl::SleepFor(simulated_work_);
    }
    ::litert::omni::TextOutput out;
    out.confirmed_text = std::to_string(*val);
    PushOutput(std::move(out));
    return absl::OkStatus();
  }

 private:
  ::litert::omni::Stage<int64_t>& upstream_;
  absl::Duration simulated_work_;
};

}  // namespace

class MicrobenchTransformCalculator : public mediapipe::CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int64_t>();
    cc->Outputs().Index(0).Set<int64_t>();
    if (cc->InputSidePackets().HasTag("WORK_MS")) {
      cc->InputSidePackets().Tag("WORK_MS").Set<int>();
    }
    return absl::OkStatus();
  }
  absl::Status Open(mediapipe::CalculatorContext* cc) override {
    cc->SetOffset(mediapipe::TimestampDiff(0));
    if (cc->InputSidePackets().HasTag("WORK_MS")) {
      work_ms_ = cc->InputSidePackets().Tag("WORK_MS").Get<int>();
    }
    return absl::OkStatus();
  }
  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    int64_t val = cc->Inputs().Index(0).Get<int64_t>();
    if (work_ms_ > 0) {
      absl::SleepFor(absl::Milliseconds(work_ms_));
    }
    cc->Outputs().Index(0).AddPacket(
        mediapipe::MakePacket<int64_t>(val * 2 + 1).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }

 private:
  int work_ms_ = 0;
};
REGISTER_CALCULATOR(
    ::mediapipe::tasks::audio::benchmarks::MicrobenchTransformCalculator);

namespace {

struct SchedulerBenchmarkResult {
  double elapsed_ms = 0.0;
  double first_packet_ms = 0.0;
  int extra_threads = 0;
  int packets_received = 0;
};

absl::StatusOr<SchedulerBenchmarkResult> RunAsyncStageSchedulerBenchmark(
    int num_packets, absl::Duration s1_work = absl::ZeroDuration(),
    absl::Duration s2_work = absl::ZeroDuration()) {
  SchedulerBenchmarkResult result;
  int threads_before = GetCurrentProcessThreadCount();
  auto pool = std::make_unique<::litert::lm::ThreadPool>("bench_omni_pool",
                                                         kDefaultCpuThreads);
  auto s1 = std::make_unique<MicrobenchSourceStage>(num_packets);
  auto s2 = std::make_unique<MicrobenchTransformStage>(
      static_cast<::litert::omni::Stage<int64_t>&>(*s1), s1_work);
  auto s3 = std::make_unique<MicrobenchTransformStage>(
      static_cast<::litert::omni::Stage<int64_t>&>(*s2), s2_work);
  std::vector<::litert::omni::internal::StageBase*> raw_stages = {
      s1.get(), s2.get(), s3.get()};
  absl::Notification done;
  absl::Time t0 = absl::Now();
  ::litert::omni::AsyncStageScheduler<int64_t> scheduler(
      raw_stages, s3.get(), pool.get(),
      [&result, &done, &t0](absl::StatusOr<int64_t> res) -> absl::Status {
        if (!res.ok()) {
          if (!done.HasBeenNotified()) {
            done.Notify();
          }
          return res.status();
        }
        if (result.packets_received++ == 0) {
          result.first_packet_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
        }
        return absl::OkStatus();
      });
  result.extra_threads =
      std::max(0, GetCurrentProcessThreadCount() - threads_before);

  t0 = absl::Now();
  ABSL_RETURN_IF_ERROR(scheduler.Start());
  done.WaitForNotification();
  ABSL_RETURN_IF_ERROR(scheduler.Stop(kSchedulerStopTimeout));
  result.elapsed_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
  return result;
}

absl::StatusOr<SchedulerBenchmarkResult> RunMultiStagedSessionBenchmark(
    int num_packets, absl::Duration s1_work = absl::ZeroDuration(),
    absl::Duration s2_work = absl::ZeroDuration()) {
  SchedulerBenchmarkResult result;
  int threads_before = GetCurrentProcessThreadCount();
  auto pool = std::make_unique<::litert::lm::ThreadPool>("bench_mss_pool",
                                                         kDefaultCpuThreads);
  auto s1 = std::make_unique<MicrobenchSourceStage>(num_packets);
  auto s2 = std::make_unique<MicrobenchTransformStage>(*s1, s1_work);
  auto s3 = std::make_unique<MicrobenchSinkStage>(*s2, s2_work);
  auto* sink_ptr = s3.get();
  std::vector<std::unique_ptr<::litert::omni::internal::StageBase>> stages;
  stages.push_back(std::move(s1));
  stages.push_back(std::move(s2));
  stages.push_back(std::move(s3));
  ABSL_ASSIGN_OR_RETURN(
      auto session,
      MultiStagedSession::Create(std::move(stages), sink_ptr, pool.get()));
  result.extra_threads =
      std::max(0, GetCurrentProcessThreadCount() - threads_before);

  absl::Time t0 = absl::Now();
  while (result.packets_received < num_packets) {
    ABSL_RETURN_IF_ERROR(session->ProcessNext().status());
    if (result.packets_received++ == 0) {
      result.first_packet_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
    }
  }
  result.elapsed_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
  return result;
}

absl::StatusOr<SchedulerBenchmarkResult> RunCalculatorGraphBenchmark(
    int num_packets, int s1_work_ms = 0, int s2_work_ms = 0,
    int num_threads = 0) {
  constexpr absl::string_view kCalcName =
      "mediapipe.tasks.audio.benchmarks.MicrobenchTransformCalculator";
  mediapipe::CalculatorGraphConfig config;
  if (num_threads <= 0) {
    auto* executor = config.add_executor();
    executor->set_name("");
    executor->set_type("ApplicationThreadExecutor");
  } else {
    config.set_num_threads(num_threads);
  }
  config.add_input_stream("in");
  auto* n1 = config.add_node();
  n1->set_calculator(kCalcName);
  n1->add_input_stream("in");
  n1->add_input_side_packet("WORK_MS:s1_ms");
  n1->add_output_stream("mid1");
  auto* n2 = config.add_node();
  n2->set_calculator(kCalcName);
  n2->add_input_stream("mid1");
  n2->add_input_side_packet("WORK_MS:s2_ms");
  n2->add_output_stream("out");

  SchedulerBenchmarkResult result;
  int threads_before = GetCurrentProcessThreadCount();
  mediapipe::CalculatorGraph graph;
  ABSL_RETURN_IF_ERROR(graph.Initialize(config));
  absl::Time t0 = absl::Now();
  ABSL_RETURN_IF_ERROR(graph.ObserveOutputStream(
      "out", [&result, &t0](const mediapipe::Packet& p) {
        (void)p.Get<int64_t>();
        if (result.packets_received++ == 0) {
          result.first_packet_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
        }
        return absl::OkStatus();
      }));
  ABSL_RETURN_IF_ERROR(
      graph.StartRun({{"s1_ms", mediapipe::MakePacket<int>(s1_work_ms)},
                      {"s2_ms", mediapipe::MakePacket<int>(s2_work_ms)}}));
  result.extra_threads =
      std::max(0, GetCurrentProcessThreadCount() - threads_before);

  t0 = absl::Now();
  for (int i = 1; i <= num_packets; ++i) {
    ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in", mediapipe::MakePacket<int64_t>(i).At(mediapipe::Timestamp(i))));
  }
  ABSL_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  ABSL_RETURN_IF_ERROR(graph.WaitUntilDone());
  result.elapsed_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
  return result;
}

TEST(SchedulerVsMediaPipeBenchmarkTest,
     AsyncStageSchedulerVsCalculatorGraphLatencyAndThreads) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(SchedulerBenchmarkResult async_res,
                               RunAsyncStageSchedulerBenchmark(kNumPackets));
  EXPECT_EQ(async_res.packets_received, kNumPackets);

  ABSL_MP_ASSERT_OK_AND_ASSIGN(SchedulerBenchmarkResult mss_res,
                               RunMultiStagedSessionBenchmark(kNumPackets));
  EXPECT_EQ(mss_res.packets_received, kNumPackets);

  ABSL_MP_ASSERT_OK_AND_ASSIGN(SchedulerBenchmarkResult mp_res,
                               RunCalculatorGraphBenchmark(kNumPackets));
  EXPECT_EQ(mp_res.packets_received, kNumPackets);

  std::cout << "\n============================================================"
               "==================\n"
            << "PART 1A: ZERO-WORK SCHEDULING OVERHEAD (" << kNumPackets
            << " packets)\n"
            << "============================================================"
               "==================\n"
            << absl::StrFormat(
                   "| %-34s | %10.2f ms | %8.2f us/pkt | +%d pool threads |\n",
                   "LiteRT-Omni AsyncStageScheduler", async_res.elapsed_ms,
                   (async_res.elapsed_ms * kUsPerMillisecond) / kNumPackets,
                   async_res.extra_threads)
            << absl::StrFormat(
                   "| %-34s | %10.2f ms | %8.2f us/pkt | +%d idle threads |\n",
                   "LiteRT-Omni MultiStagedSession", mss_res.elapsed_ms,
                   (mss_res.elapsed_ms * kUsPerMillisecond) / kNumPackets,
                   mss_res.extra_threads)
            << absl::StrFormat(
                   "| %-34s | %10.2f ms | %8.2f us/pkt | 0 extra threads |\n",
                   "MediaPipe CalculatorGraph (1T)", mp_res.elapsed_ms,
                   (mp_res.elapsed_ms * kUsPerMillisecond) / kNumPackets)
            << "============================================================"
               "==================\n";
}

TEST(SchedulerVsMediaPipeBenchmarkTest,
     MultiStagePipelineOverlapVsOmniSequentialAndStall) {
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      SchedulerBenchmarkResult mss_res,
      RunMultiStagedSessionBenchmark(kPipelineChunks, kSimulatedStage1Work,
                                     kSimulatedStage2Work));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      SchedulerBenchmarkResult async_res,
      RunAsyncStageSchedulerBenchmark(kPipelineChunks, kSimulatedStage1Work,
                                      kSimulatedStage2Work));
  ABSL_MP_ASSERT_OK_AND_ASSIGN(
      SchedulerBenchmarkResult mp_res,
      RunCalculatorGraphBenchmark(
          kPipelineChunks,
          static_cast<int>(absl::ToInt64Milliseconds(kSimulatedStage1Work)),
          static_cast<int>(absl::ToInt64Milliseconds(kSimulatedStage2Work)),
          /*num_threads=*/2));

  EXPECT_EQ(mss_res.packets_received, kPipelineChunks);
  EXPECT_EQ(async_res.packets_received, kPipelineChunks);
  EXPECT_EQ(mp_res.packets_received, kPipelineChunks);
  EXPECT_LT(mp_res.elapsed_ms, mss_res.elapsed_ms * 0.80);
  EXPECT_GT(async_res.elapsed_ms, 900.0);

  std::cout
      << "\n============================================================"
         "==================\n"
      << "PART 1B: MULTI-STAGE PIPELINE OVERLAP (" << kPipelineChunks
      << " chunks: 12ms Stage1 + 18ms Stage2)\n"
      << "============================================================"
         "==================\n"
      << absl::StrFormat(
             "| %-34s | Total: %7.1f ms | TTFA: %6.1f ms | Sequential |\n",
             "LiteRT-Omni MultiStagedSession", mss_res.elapsed_ms,
             mss_res.first_packet_ms)
      << absl::StrFormat(
             "| %-34s | Total: %7.1f ms | TTFA: %6.1f ms | 1s Stall   |\n",
             "LiteRT-Omni AsyncStageScheduler", async_res.elapsed_ms,
             async_res.first_packet_ms)
      << absl::StrFormat(
             "| %-34s | Total: %7.1f ms | TTFA: %6.1f ms | Pipelined  |\n",
             "MediaPipe CalculatorGraph (2T)", mp_res.elapsed_ms,
             mp_res.first_packet_ms)
      << "============================================================"
         "==================\n";
}

TEST(SchedulerVsMediaPipeBenchmarkTest,
     ComputeMultilingualErrorRateNormalizesScriptsAndDigits) {
  EXPECT_DOUBLE_EQ(
      ComputeMultilingualErrorRate("Hello, World!", "hello world", "en-US"),
      0.0);
  EXPECT_DOUBLE_EQ(ComputeMultilingualErrorRate("", "", "ja-JP"), 0.0);
  EXPECT_DOUBLE_EQ(ComputeMultilingualErrorRate("", "abc", "ja-JP"), 1.0);
  // Full-width digits, CJK numerals, CJK punctuation, 2-byte and 4-byte UTF-8.
  EXPECT_DOUBLE_EQ(ComputeMultilingualErrorRate(
                       "零〇一二两三四五六七八九、。！？"
                       "\xEF\xBC\x91\xEF\xBC\x92\xC3\xA9\xF0\x9F\x98\x80",
                       "00122345678912\xC3\xA9\xF0\x9F\x98\x80", "zh-CN"),
                   0.0);
  // Devanagari digits and danda/nukta.
  EXPECT_DOUBLE_EQ(
      ComputeMultilingualErrorRate(
          "\xE0\xA5\xA7\xE0\xA5\xA8\xE0\xA5\xA4\xE0\xA4\xBC", "12", "hi-IN"),
      0.0);
  EXPECT_GT(ComputeMultilingualErrorRate("一二三", "一二四", "ja-JP"), 0.0);
}

}  // namespace
}  // namespace mediapipe::tasks::audio::benchmarks
