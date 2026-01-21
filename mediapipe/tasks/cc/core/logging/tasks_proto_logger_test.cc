#include "mediapipe/tasks/cc/core/logging/tasks_proto_logger.h"

#include <memory>
#include <utility>

#include "mediapipe/tasks/cc/core/logging/logging_client.h"
#include "mediapipe/util/analytics/mediapipe_log_extension.pb.h"
#include "mediapipe/util/analytics/mediapipe_logging_enums.pb.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace logging {
namespace {

using logs::proto::mediapipe::MediaPipeLogExtension;
using logs::proto::mediapipe::Platform;
using ::testing::EqualsProto;
using ::testing::proto::Partially;

// Mock class for LoggingClient to intercept and verify logged events.
class MockLoggingClient : public LoggingClient {
 public:
  MOCK_METHOD(void, LogEvent, (const MediaPipeLogExtension& event), (override));
};

class TasksStatsProtoLoggerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_logging_client_ = std::make_unique<MockLoggingClient>();
    mock_logging_client_ptr_ = mock_logging_client_.get();
  }

  std::unique_ptr<MockLoggingClient> mock_logging_client_;
  MockLoggingClient* mock_logging_client_ptr_;
};

TEST_F(TasksStatsProtoLoggerTest, LogSessionStart) {
  auto logger = TasksStatsProtoLogger::Create(
      "test_app", "1.0.0", "ImageClassifier", "live_stream",
      std::move(mock_logging_client_), Platform::PLATFORM_ANDROID);

  MediaPipeLogExtension expected_log;
  auto* system_info = expected_log.mutable_system_info();
  system_info->set_app_id("test_app");
  system_info->set_app_version("1.0.0");
  system_info->set_platform(Platform::PLATFORM_ANDROID);

  auto* solution_event = expected_log.mutable_solution_event();
  solution_event->set_solution_name(
      logs::proto::mediapipe::SolutionName::TASKS_IMAGECLASSIFIER);
  solution_event->set_event_name(
      logs::proto::mediapipe::EventName::EVENT_START);
  solution_event->mutable_session_start()->set_mode(
      logs::proto::mediapipe::SolutionMode::MODE_TASKS_LIVE_STREAM);
  // init_latency_ms is time-dependent, so we use Partially.

  EXPECT_CALL(*mock_logging_client_ptr_,
              LogEvent(Partially(EqualsProto(expected_log))))
      .Times(1);

  logger->LogSessionStart();
}

TEST_F(TasksStatsProtoLoggerTest, LogSessionClone) {
  auto logger = TasksStatsProtoLogger::Create(
      "test_app", "1.0.0", "LlmInference", "synchronous",
      std::move(mock_logging_client_), Platform::PLATFORM_IOS);

  MediaPipeLogExtension expected_log;
  auto* system_info = expected_log.mutable_system_info();
  system_info->set_app_id("test_app");
  system_info->set_app_version("1.0.0");
  system_info->set_platform(Platform::PLATFORM_IOS);

  auto* solution_event = expected_log.mutable_solution_event();
  solution_event->set_solution_name(
      logs::proto::mediapipe::SolutionName::TASKS_LLMINFERENCE);
  solution_event->set_event_name(
      logs::proto::mediapipe::EventName::EVENT_START);

  EXPECT_CALL(*mock_logging_client_ptr_,
              LogEvent(Partially(EqualsProto(expected_log))))
      .Times(1);

  logger->LogSessionClone();
}

TEST_F(TasksStatsProtoLoggerTest, LogInitError) {
  auto logger = TasksStatsProtoLogger::Create(
      "test_app", "1.0.0", "ObjectDetector", "image",
      std::move(mock_logging_client_), Platform::PLATFORM_ANDROID);

  MediaPipeLogExtension expected_log;
  auto* system_info = expected_log.mutable_system_info();
  system_info->set_app_id("test_app");
  system_info->set_app_version("1.0.0");
  system_info->set_platform(Platform::PLATFORM_ANDROID);

  auto* solution_event = expected_log.mutable_solution_event();
  solution_event->set_solution_name(
      logs::proto::mediapipe::SolutionName::TASKS_OBJECTDETECTOR);
  solution_event->set_event_name(
      logs::proto::mediapipe::EventName::EVENT_ERROR);
  solution_event->mutable_error_details()->set_error_code(
      logs::proto::mediapipe::ErrorCode::ERROR_INIT);

  EXPECT_CALL(*mock_logging_client_ptr_, LogEvent(EqualsProto(expected_log)))
      .Times(1);

  logger->LogInitError();
}

TEST_F(TasksStatsProtoLoggerTest, LogSessionEndWithInvocationStats) {
  auto logger = TasksStatsProtoLogger::Create(
      "test_app", "1.0.0", "FaceLandmarker", "video",
      std::move(mock_logging_client_), Platform::PLATFORM_ANDROID);

  // Start a session.
  logger->LogSessionStart();

  // Simulate some invocations, including one dropped packet.
  logger->RecordCpuInputArrival(/*packet_timestamp=*/100);
  logger->RecordInvocationEnd(/*packet_timestamp=*/100);

  logger->RecordGpuInputArrival(/*packet_timestamp=*/200);
  logger->RecordInvocationEnd(/*packet_timestamp=*/200);

  logger->RecordCpuInputArrival(/*packet_timestamp=*/300);

  MediaPipeLogExtension expected_log;
  auto* system_info = expected_log.mutable_system_info();
  system_info->set_app_id("test_app");
  system_info->set_app_version("1.0.0");
  system_info->set_platform(Platform::PLATFORM_ANDROID);

  auto* solution_event = expected_log.mutable_solution_event();
  solution_event->set_solution_name(
      logs::proto::mediapipe::SolutionName::TASKS_FACELANDMARKER);
  solution_event->set_event_name(logs::proto::mediapipe::EventName::EVENT_END);
  auto* session_end = solution_event->mutable_session_end();
  auto* invocation_report = session_end->mutable_invocation_report();
  invocation_report->set_mode(
      logs::proto::mediapipe::SolutionMode::MODE_TASKS_VIDEO);
  invocation_report->set_dropped(1);  // One dropped packet (timestamp 300)

  auto* cpu_count = invocation_report->add_invocation_count();
  cpu_count->set_input_data_type(
      logs::proto::mediapipe::InputDataType::INPUT_TYPE_TASKS_CPU);
  cpu_count->set_count(2);  // One from 100, one from 300

  auto* gpu_count = invocation_report->add_invocation_count();
  gpu_count->set_input_data_type(
      logs::proto::mediapipe::InputDataType::INPUT_TYPE_TASKS_GPU);
  gpu_count->set_count(1);  // One from 200

  EXPECT_CALL(*mock_logging_client_ptr_,
              LogEvent(Partially(EqualsProto(expected_log))))
      .Times(1);

  logger->LogSessionEnd();
}

}  // namespace
}  // namespace logging
}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
