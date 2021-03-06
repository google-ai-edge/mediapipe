#include "mediapipe/framework/api2/contract.h"

#include <tuple>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace api2 {
namespace {

struct ProcessItem {
  absl::Status Process(CalculatorContext* cc) { return {}; }
};

struct ItemWithNested {
  constexpr auto nested_items() { return std::make_tuple(Input<char>{"FWD"}); }
};

static constexpr auto kTestContract = internal::MakeContract(
    Input<int>{"BASE"}, Input<float>::Optional{"SCALE"}, Output<float>{"OUT"},
    SideInput<float>::Optional{"BIAS"}, SideOutput<char>{"SIDE"},
    ProcessItem{});

static_assert(std::tuple_size_v<decltype(kTestContract.inputs())> == 2, "");
static_assert(std::tuple_size_v<decltype(kTestContract.outputs())> == 1, "");
static_assert(std::tuple_size_v<decltype(kTestContract.side_inputs())> == 1,
              "");
static_assert(std::tuple_size_v<decltype(kTestContract.side_outputs())> == 1,
              "");

static_assert(internal::HasProcessMethod<ProcessItem>{}, "");
static_assert(!internal::HasProcessMethod<Input<int>>{}, "");

static_assert(std::tuple_size_v<decltype(kTestContract.process_items())> == 1,
              "");

static constexpr auto kExtractNested1 = internal::ExtractNestedItems(
    std::make_tuple(Input<int>{"BASE"}, Input<float>::Optional{"SCALE"},
                    Output<float>{"OUT"}));

static_assert(std::tuple_size_v<decltype(kExtractNested1)> == 3, "");

static constexpr auto kExtractNested2 = internal::ExtractNestedItems(
    std::make_tuple(Input<int>{"BASE"}, Input<float>::Optional{"SCALE"},
                    Output<float>{"OUT"}, ItemWithNested{}));
static_assert(std::tuple_size_v<decltype(kExtractNested2)> == 5, "");

using TaggedTestContract =
    internal::TaggedContract<decltype(kTestContract), kTestContract>;

static constexpr auto kBASE = MPP_TAG("BASE");
static constexpr auto kSCALE = MPP_TAG("SCALE");
static constexpr auto kBIAS = MPP_TAG("BIAS");
static constexpr auto kOUT = MPP_TAG("OUT");
static constexpr auto kSIDE = MPP_TAG("SIDE");

static_assert(TaggedTestContract::TaggedInputs::get(kBASE).tag_ == kBASE.kStr,
              "");
static_assert(TaggedTestContract::TaggedInputs::get(kSCALE).tag_ == kSCALE.kStr,
              "");
static_assert(TaggedTestContract::TaggedOutputs::get(kOUT).tag_ == kOUT.kStr,
              "");
static_assert(TaggedTestContract::TaggedSideInputs::get(kBIAS).tag_ ==
                  kBIAS.kStr,
              "");
static_assert(TaggedTestContract::TaggedSideOutputs::get(kSIDE).tag_ ==
                  kSIDE.kStr,
              "");

}  // namespace
}  // namespace api2
}  // namespace mediapipe
