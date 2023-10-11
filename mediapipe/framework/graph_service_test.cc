// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/graph_service.h"

#include <type_traits>

#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/test_service.h"

namespace mediapipe {

namespace {

using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::UnorderedElementsAre;

// Returns the packet values for a vector of Packets.
template <typename T>
std::vector<T> PacketValues(const std::vector<Packet>& packets) {
  std::vector<T> result;
  for (const Packet& packet : packets) {
    result.push_back(packet.Get<T>());
  }
  return result;
}

class GraphServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "in"
          node {
            calculator: "TestServiceCalculator"
            input_stream: "in"
            output_stream: "out"
          }
        )pb");
    MP_ASSERT_OK(graph_.Initialize(config));
    MP_ASSERT_OK(
        graph_.ObserveOutputStream("out", [this](const Packet& packet) {
          output_packets_.push_back(packet);
          return absl::OkStatus();
        }));
  }

  CalculatorGraph graph_;
  std::vector<Packet> output_packets_;
};

TEST_F(GraphServiceTest, SetOnGraph) {
  EXPECT_EQ(graph_.GetServiceObject(kTestService).get(), nullptr);
  auto service_object =
      std::make_shared<TestServiceObject>(TestServiceObject{{"delta", 3}});
  MP_EXPECT_OK(graph_.SetServiceObject(kTestService, service_object));
  EXPECT_EQ(graph_.GetServiceObject(kTestService), service_object);

  service_object = std::make_shared<TestServiceObject>(
      TestServiceObject{{"delta", 5}, {"count", 0}});

  MP_EXPECT_OK(graph_.SetServiceObject(kTestService, service_object));
  EXPECT_EQ(graph_.GetServiceObject(kTestService), service_object);
}

TEST_F(GraphServiceTest, UseInCalculator) {
  auto service_object = std::make_shared<TestServiceObject>(
      TestServiceObject{{"delta", 5}, {"count", 0}});
  MP_EXPECT_OK(graph_.SetServiceObject(kTestService, service_object));

  MP_ASSERT_OK(graph_.StartRun({}));
  MP_ASSERT_OK(
      graph_.AddPacketToInputStream("in", MakePacket<int>(3).At(Timestamp(0))));
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());
  EXPECT_EQ(PacketValues<int>(output_packets_), (std::vector<int>{8}));
  EXPECT_EQ(1, (*service_object)["count"]);
}

TEST_F(GraphServiceTest, Contract) {
  const CalculatorGraphConfig::Node node =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TestServiceCalculator"
        input_stream: "in"
        output_stream: "out"
      )pb");
  CalculatorContract contract;
  MP_EXPECT_OK(contract.Initialize(node));
  MP_EXPECT_OK(TestServiceCalculator::GetContract(&contract));
  EXPECT_THAT(
      contract.ServiceRequests(),
      UnorderedElementsAre(Key(kTestService.key), Key(kAnotherService.key)));
  EXPECT_EQ(contract.ServiceRequests().at(kTestService.key).IsOptional(),
            false);
  EXPECT_EQ(contract.ServiceRequests().at(kAnotherService.key).IsOptional(),
            true);
}

TEST_F(GraphServiceTest, MustHaveRequired) {
  // Do not provide any service.
  auto status = graph_.StartRun({});
  EXPECT_THAT(status.message(), HasSubstr(kTestService.key));
}

TEST_F(GraphServiceTest, OptionalIsOptional) {
  // Provide only required service.
  auto service_object = std::make_shared<TestServiceObject>(
      TestServiceObject{{"delta", 5}, {"count", 0}});
  MP_EXPECT_OK(graph_.SetServiceObject(kTestService, service_object));

  MP_EXPECT_OK(graph_.StartRun({}));
  MP_ASSERT_OK(
      graph_.AddPacketToInputStream("in", MakePacket<int>(3).At(Timestamp(0))));
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());
  EXPECT_EQ(PacketValues<int>(output_packets_), (std::vector<int>{8}));
}

TEST_F(GraphServiceTest, OptionalIsAvailable) {
  auto service_object = std::make_shared<TestServiceObject>(
      TestServiceObject{{"delta", 5}, {"count", 0}});
  MP_EXPECT_OK(graph_.SetServiceObject(kTestService, service_object));
  MP_EXPECT_OK(
      graph_.SetServiceObject(kAnotherService, std::make_shared<int>(100)));

  MP_EXPECT_OK(graph_.StartRun({}));
  MP_ASSERT_OK(
      graph_.AddPacketToInputStream("in", MakePacket<int>(3).At(Timestamp(0))));
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());
  EXPECT_EQ(PacketValues<int>(output_packets_), (std::vector<int>{108}));
}

TEST_F(GraphServiceTest, CreateDefault) {
  EXPECT_FALSE(kTestService.CreateDefaultObject().ok());
  MP_EXPECT_OK(kAnotherService.CreateDefaultObject());
  EXPECT_FALSE(kNoDefaultService.CreateDefaultObject().ok());
  MP_EXPECT_OK(kNeedsCreateService.CreateDefaultObject());
}

struct TestServiceData {};

constexpr GraphService<TestServiceData> kTestServiceAllowDefaultInitialization(
    "kTestServiceAllowDefaultInitialization",
    GraphServiceBase::kAllowDefaultInitialization);

// This is only for test purposes. Ideally, a calculator that fails when service
// is not available, should request the service as non-Optional.
class FailOnUnavailableOptionalServiceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->UseService(kTestServiceAllowDefaultInitialization).Optional();
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    RET_CHECK(cc->Service(kTestServiceAllowDefaultInitialization).IsAvailable())
        << "Service is unavailable.";
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }
};
REGISTER_CALCULATOR(FailOnUnavailableOptionalServiceCalculator);

// Documents and ensures current behavior for requesting optional
// "AllowDefaultInitialization" services:
// - Service object is created by default.
TEST(AllowDefaultInitializationGraphServiceTest,
     ServiceIsAvailableWithOptionalUse) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node { calculator: 'FailOnUnavailableOptionalServiceCalculator' }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.WaitUntilIdle());
}

// Documents and ensures current behavior for setting `nullptr` service objects
// for "AllowDefaultInitialization" optional services.
// - It's allowed.
// - It disables creation of "AllowDefaultInitialization" service objects, hence
//   results in optional service unavailability.
TEST(AllowDefaultInitializationGraphServiceTest,
     NullServiceObjectIsAllowAndResultsInOptionalServiceUnavailability) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node { calculator: 'FailOnUnavailableOptionalServiceCalculator' }
      )pb");

  CalculatorGraph graph;
  std::shared_ptr<TestServiceData> object = nullptr;
  MP_ASSERT_OK(
      graph.SetServiceObject(kTestServiceAllowDefaultInitialization, object));
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  EXPECT_THAT(graph.WaitUntilIdle(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Service is unavailable.")));
}

class FailOnUnavailableServiceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->UseService(kTestServiceAllowDefaultInitialization);
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    RET_CHECK(cc->Service(kTestServiceAllowDefaultInitialization).IsAvailable())
        << "Service is unavailable.";
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }
};
REGISTER_CALCULATOR(FailOnUnavailableServiceCalculator);

// Documents and ensures current behavior for requesting optional
// "AllowDefaultInitialization" services:
// - Service object is created by default.
TEST(AllowDefaultInitializationGraphServiceTest, ServiceIsAvailable) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node { calculator: 'FailOnUnavailableServiceCalculator' }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.WaitUntilIdle());
}

// Documents and ensures current behavior for setting `nullptr` service objects
// for "AllowDefaultInitialization" services.
// - It's allowed.
// - It disables creation of "AllowDefaultInitialization" service objects, hence
//   in service unavaialbility.
TEST(AllowDefaultInitializationGraphServiceTest,
     NullServiceObjectIsAllowAndResultsInServiceUnavailability) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node { calculator: 'FailOnUnavailableServiceCalculator' }
      )pb");

  CalculatorGraph graph;
  std::shared_ptr<TestServiceData> object = nullptr;
  MP_ASSERT_OK(
      graph.SetServiceObject(kTestServiceAllowDefaultInitialization, object));
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  EXPECT_THAT(graph.WaitUntilIdle(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Service is unavailable.")));
}

constexpr GraphService<TestServiceData>
    kTestServiceDisallowDefaultInitialization(
        "kTestServiceDisallowDefaultInitialization",
        GraphServiceBase::kDisallowDefaultInitialization);

static_assert(std::is_trivially_destructible_v<GraphService<TestServiceData>>,
              "GraphService is not trivially destructible");

class FailOnUnavailableOptionalDisallowDefaultInitServiceCalculator
    : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->UseService(kTestServiceDisallowDefaultInitialization).Optional();
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    RET_CHECK(
        cc->Service(kTestServiceDisallowDefaultInitialization).IsAvailable())
        << "Service is unavailable.";
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }
};
REGISTER_CALCULATOR(
    FailOnUnavailableOptionalDisallowDefaultInitServiceCalculator);

// Documents and ensures current behavior for requesting optional
// "DisallowDefaultInitialization" services:
// - Service object is not created by default.
TEST(DisallowDefaultInitializationGraphServiceTest,
     ServiceIsUnavailableWithOptionalUse) {
  CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<
      CalculatorGraphConfig>(R"pb(
    node {
      calculator: 'FailOnUnavailableOptionalDisallowDefaultInitServiceCalculator'
    }
  )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  EXPECT_THAT(graph.WaitUntilIdle(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Service is unavailable.")));
}

class UseDisallowDefaultInitServiceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->UseService(kTestServiceDisallowDefaultInitialization);
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }
  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }
};
REGISTER_CALCULATOR(UseDisallowDefaultInitServiceCalculator);

// Documents and ensures current behavior for requesting
// "DisallowDefaultInitialization" services:
// - Service object is not created by default.
// - Graph run fails.
TEST(DisallowDefaultInitializationGraphServiceTest,
     StartRunFailsMissingService) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node { calculator: 'UseDisallowDefaultInitServiceCalculator' }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  EXPECT_THAT(graph.StartRun({}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("was not provided and cannot be created")));
}

TEST(ServiceBindingTest, CrashesWhenGettingNullServiceObject) {
  ASSERT_DEATH(
      {
        ServiceBinding<TestServiceData> binding(nullptr);
        (void)binding.GetObject();
      },
      testing::ContainsRegex("Check failed: [a-z_]* Service is unavailable"));
}

TEST(ServiceBindingTest, IsAvailableReturnsFalsOnNullServiceObject) {
  ServiceBinding<TestServiceData> binding(nullptr);
  EXPECT_FALSE(binding.IsAvailable());
}

}  // namespace
}  // namespace mediapipe
