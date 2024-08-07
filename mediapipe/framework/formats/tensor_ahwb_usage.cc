
#include "mediapipe/framework/formats/tensor_ahwb_usage.h"

#include <list>

#include "absl/log/absl_log.h"

namespace mediapipe {

void EraseCompletedUsages(std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto it = ahwb_usages.begin(); it != ahwb_usages.end();) {
    bool is_ready = true;
    if (it->is_complete_fn) {
      if (!it->is_complete_fn(/*force_completion=*/false)) {
        is_ready = false;
      }
    } else {
      ABSL_LOG(ERROR) << "Usage is missing completion function.";
    }

    if (is_ready) {
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
    if (ahwb_usage.is_complete_fn &&
        !ahwb_usage.is_complete_fn(/*force_completion=*/false)) {
      if (!ahwb_usage.is_complete_fn(/*force_completion=*/true)) {
        ABSL_LOG(DFATAL) << "Failed to force-complete AHWB usage.";
      }
    }

    for (auto& release_callback : ahwb_usage.release_callbacks) {
      release_callback();
    }
  }
  ahwb_usages.clear();
}

bool HasIncompleteUsages(const std::list<TensorAhwbUsage>& ahwb_usages) {
  for (auto& ahwb_usage : ahwb_usages) {
    if (ahwb_usage.is_complete_fn &&
        !ahwb_usage.is_complete_fn(/*force_completion=*/false)) {
      return true;
    }
  }
  return false;
}

}  // namespace mediapipe
