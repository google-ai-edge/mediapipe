#include "mediapipe/framework/formats/tensor_ahwb_usage.h"

#include <list>
#include <utility>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

namespace {

TEST(HardwareBufferTest, ShouldDetectHasIncompleteUsage) {
  TensorAhwbUsage ahwb_usage;
  bool requested_force_completion = false;
  ahwb_usage.is_complete_fn = [&](bool force_completion) {
    requested_force_completion = force_completion;
    return false;
  };
  EXPECT_FALSE(requested_force_completion);
  EXPECT_TRUE(HasIncompleteUsage(ahwb_usage));
}

TEST(HardwareBufferTest, ShouldNotDetectHasIncompleteUsage) {
  TensorAhwbUsage ahwb_usage;
  bool requested_force_completion = false;
  ahwb_usage.is_complete_fn = [&](bool force_completion) {
    requested_force_completion = force_completion;
    return true;
  };
  EXPECT_FALSE(requested_force_completion);
  EXPECT_FALSE(HasIncompleteUsage(ahwb_usage));
}

TEST(HardwareBufferTest, ShouldDetectHasIncompleteUsageFromList) {
  TensorAhwbUsage ahwb_usage;
  ahwb_usage.is_complete_fn = [&](bool force_completion) { return false; };
  std::list<TensorAhwbUsage> ahwb_usages;
  ahwb_usages.push_back(std::move(ahwb_usage));
  EXPECT_TRUE(HasIncompleteUsages(ahwb_usages));
}

TEST(HardwareBufferTest, ShouldNotDetectHasIncompleteUsageFromList) {
  TensorAhwbUsage ahwb_usage;
  ahwb_usage.is_complete_fn = [&](bool force_completion) { return true; };
  std::list<TensorAhwbUsage> ahwb_usages;
  ahwb_usages.push_back(std::move(ahwb_usage));
  EXPECT_FALSE(HasIncompleteUsages(ahwb_usages));
}

TEST(HardwareBufferTest, ShouldForceCompleteUsage) {
  TensorAhwbUsage ahwb_usage;
  bool requested_force_completion = false;
  ahwb_usage.is_complete_fn = [&](bool force_completion) {
    requested_force_completion |= force_completion;
    return requested_force_completion;
  };
  CompleteAndEraseUsage(ahwb_usage);
  EXPECT_TRUE(requested_force_completion);
}

TEST(HardwareBufferTest, ShouldForceCompleteAndEraseUsageFromList) {
  TensorAhwbUsage ahwb_usage;
  bool requested_force_completion = false;
  ahwb_usage.is_complete_fn = [&](bool force_completion) {
    requested_force_completion |= force_completion;
    return requested_force_completion;
  };
  std::list<TensorAhwbUsage> ahwb_usages;
  ahwb_usages.push_back(std::move(ahwb_usage));
  CompleteAndEraseUsages(ahwb_usages);
  EXPECT_TRUE(requested_force_completion);
  EXPECT_TRUE(ahwb_usages.empty());
}

}  // namespace

}  // namespace mediapipe
