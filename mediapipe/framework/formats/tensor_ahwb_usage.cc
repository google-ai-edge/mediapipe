
#include "mediapipe/framework/formats/tensor_ahwb_usage.h"

#include <list>

#include "absl/log/absl_log.h"

namespace mediapipe {

bool TensorAhwbUsage::IsComplete() const {
  if (is_complete_fn != nullptr &&
      !is_complete_fn(/*force_completion=*/false)) {
    return false;
  }
  return true;
}

void TensorAhwbUsage::Reset() {
  if (is_complete_fn != nullptr &&
      !is_complete_fn(/*force_completion=*/false) &&
      !is_complete_fn(/*force_completion=*/true)) {
    ABSL_LOG(DFATAL) << "Failed to force-complete AHWB usage.";
  }
  for (auto& release_callback : release_callbacks) {
    release_callback();
  }
  is_complete_fn = nullptr;
  release_callbacks.clear();
}

bool HasIncompleteUsages(const std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto& ahwb_usage : ahwb_usages) {
    if (!ahwb_usage.IsComplete()) {
      return true;
    }
  }
  return false;
}

void EraseCompletedUsages(std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto it = ahwb_usages.begin(); it != ahwb_usages.end();) {
    if (it->IsComplete()) {
      for (auto& release_callback : it->release_callbacks) {
        release_callback();
      }
      it = ahwb_usages.erase(it);
    } else {
      ++it;
    }
  }
}

void CompleteAndEraseUsages(std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto& ahwb_usage : ahwb_usages) {
    ahwb_usage.Reset();
  }
  ahwb_usages.clear();
}

}  // namespace mediapipe
