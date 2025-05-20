#include "mediapipe/framework/api3/internal/has_update_contract.h"

#include "absl/status/status.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::api3 {
namespace {

struct ContractType {};

struct WithUpdateContract {
  static absl::Status UpdateContract(ContractType&) {
    return absl::OkStatus();
  };
};

struct WithOutUpdateContract {};

TEST(HasUpdateContractTest, CanIdentifyUpdateContract) {
  EXPECT_TRUE((kHasUpdateContract<WithUpdateContract, ContractType>));
  EXPECT_FALSE((kHasUpdateContract<WithOutUpdateContract, ContractType>));
}

}  // namespace
}  // namespace mediapipe::api3
