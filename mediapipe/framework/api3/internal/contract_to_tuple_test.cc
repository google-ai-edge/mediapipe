#include "mediapipe/framework/api3/internal/contract_to_tuple.h"

#include <string>
#include <tuple>

#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::api3::internal_contract_to_tuple {
namespace {

template <typename S>
struct TestContract {
  Input<S, int> in_a{"A"};
  Input<S, float> in_b{"B"};
  Output<S, std::string> out{"OUT"};
};

TEST(ContractToTupleTest, CanGetFieldPtrTuple) {
  TestContract<ContractSpecializer> p;
  auto tuple = ContractToFieldPtrTuple(p);

  EXPECT_EQ(std::tuple_size_v<decltype(tuple)>, 3);
  EXPECT_EQ(std::get<0>(tuple)->Tag(), "A");
  EXPECT_EQ(std::get<1>(tuple)->Tag(), "B");
  EXPECT_EQ(std::get<2>(tuple)->Tag(), "OUT");
}

template <typename S>
struct TenPortsContract {
  Input<S, int> in_1{"1"};
  Input<S, float> in_2{"2"};
  Input<S, float> in_3{"3"};
  Input<S, float> in_4{"4"};
  Input<S, float> in_5{"5"};
  Input<S, float> in_6{"6"};
  Input<S, float> in_7{"7"};
  Input<S, float> in_8{"8"};
  Input<S, float> in_9{"9"};
  Input<S, float> in_10{"10"};
};

TEST(ContractToTupleTest, CanGetTenFieldPtrTuple) {
  TenPortsContract<ContractSpecializer> p;
  auto tuple = ContractToFieldPtrTuple(p);

  EXPECT_EQ(std::tuple_size_v<decltype(tuple)>, 10);
  EXPECT_EQ(std::get<0>(tuple)->Tag(), "1");
  EXPECT_EQ(std::get<1>(tuple)->Tag(), "2");
  EXPECT_EQ(std::get<2>(tuple)->Tag(), "3");
  EXPECT_EQ(std::get<3>(tuple)->Tag(), "4");
  EXPECT_EQ(std::get<4>(tuple)->Tag(), "5");
  EXPECT_EQ(std::get<5>(tuple)->Tag(), "6");
  EXPECT_EQ(std::get<6>(tuple)->Tag(), "7");
  EXPECT_EQ(std::get<7>(tuple)->Tag(), "8");
  EXPECT_EQ(std::get<8>(tuple)->Tag(), "9");
  EXPECT_EQ(std::get<9>(tuple)->Tag(), "10");
}

template <typename S>
struct MinPortsContract {};

TEST(ContractToTupleTest, CanGetMinFieldPtrTuple) {
  MinPortsContract<ContractSpecializer> c;
  auto tuple = ContractToFieldPtrTuple(c);

  EXPECT_EQ(std::tuple_size_v<decltype(tuple)>, 0);
}

}  // namespace
}  // namespace mediapipe::api3::internal_contract_to_tuple
