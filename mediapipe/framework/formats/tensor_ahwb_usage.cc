
#include "mediapipe/framework/formats/tensor_ahwb_usage.h"

#include <list>

#include "absl/log/absl_log.h"

namespace mediapipe {

bool HasIncompleteUsage(TensorAhwbUsage& ahwb_usage) {
  if (ahwb_usage.is_complete_fn != nullptr &&
      !ahwb_usage.is_complete_fn(/*force_completion=*/false)) {
    return true;
  }
  return false;
}

bool HasIncompleteUsages(std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto& ahwb_usage : ahwb_usages) {
    if (HasIncompleteUsage(ahwb_usage)) {
      return true;
    }
  }
  return false;
}

void EraseCompletedUsage(TensorAhwbUsage& ahwb_usage) {
  if (!HasIncompleteUsage(ahwb_usage)) {
    for (auto& release_callback : ahwb_usage.release_callbacks) {
      release_callback();
    }
    ahwb_usage.is_complete_fn = nullptr;
    ahwb_usage.release_callbacks.clear();
  }
}

void EraseCompletedUsages(std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto it = ahwb_usages.begin(); it != ahwb_usages.end();) {
    if (!HasIncompleteUsage(*it)) {
      for (auto& release_callback : it->release_callbacks) {
        release_callback();
      }
      it = ahwb_usages.erase(it);
    } else {
      ++it;
    }
  }
}

void CompleteAndEraseUsage(TensorAhwbUsage& ahwb_usage) {
  if (ahwb_usage.is_complete_fn != nullptr &&
      !ahwb_usage.is_complete_fn(/*force_completion=*/false) &&
      !ahwb_usage.is_complete_fn(/*force_completion=*/true)) {
    ABSL_LOG(DFATAL) << "Failed to force-complete AHWB usage.";
  }
  for (auto& release_callback : ahwb_usage.release_callbacks) {
    release_callback();
  }
  ahwb_usage.is_complete_fn = nullptr;
  ahwb_usage.release_callbacks.clear();
}

void CompleteAndEraseUsages(std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto& ahwb_usage : ahwb_usages) {
    CompleteAndEraseUsage(ahwb_usage);
  }
  ahwb_usages.clear();
}

}  // namespace mediapipe
