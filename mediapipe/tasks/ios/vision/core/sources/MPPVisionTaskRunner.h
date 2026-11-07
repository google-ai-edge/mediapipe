// Copyright 2023 The MediaPipe Authors.
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

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPRunningMode.h"

#ifndef MP_IMAGE_PTR_DEFINED
#define MP_IMAGE_PTR_DEFINED
typedef struct MpImageInternal *MpImagePtr;
#endif

struct MpImageProcessingOptions;

NS_ASSUME_NONNULL_BEGIN

/**
 * Shared helper for initializing and preparing inputs for MediaPipe vision C tasks.
 */
@interface MPPVisionTaskRunner : NSObject

/**
 * Initializes a new `MPPVisionTaskRunner` with running mode, capability flags, and delegate status.
 *
 * @param runningMode The task's running mode (image, video, or live stream).
 * @param roiAllowed Whether region-of-interest is supported by this task.
 * @param normRectAllowed Whether normalized rects / rotation are supported by this task.
 * @param hasDelegate Whether a result callback delegate is set on task options.
 * @param error Pointer to save error if validation fails.
 */
- (nullable instancetype)initWithRunningMode:(MPPRunningMode)runningMode
                                  roiAllowed:(BOOL)roiAllowed
                             normRectAllowed:(BOOL)normRectAllowed
                                 hasDelegate:(BOOL)hasDelegate
                                       error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Validates that the current running mode matches the expected running mode.
 *
 * @param expectedMode The expected running mode for the operation.
 * @param error Pointer to save error if validation fails.
 * @return YES if valid, NO otherwise.
 */
- (BOOL)validateRunningMode:(MPPRunningMode)expectedMode error:(NSError **)error;

/**
 * Prepares image processing options and extracts `MpImagePtr` from `MPPImage`.
 *
 * @param image The input `MPPImage`.
 * @param roi The region of interest (`CGRectZero` if not specified).
 * @param outOptions Pointer to `MpImageProcessingOptions` struct to populate.
 * @param error Pointer to save error if preparation fails.
 * @return An `MpImagePtr` handle, or `nullptr` on error.
 */
- (nullable MpImagePtr)createMpImageWithMPPImage:(MPPImage *)image
                                regionOfInterest:(CGRect)roi
                          imageProcessingOptions:(struct MpImageProcessingOptions *)outOptions
                                           error:(NSError **)error;

/**
 * This method returns a unique dispatch queue name by adding the given suffix and a `UUID` to the
 * pre-defined queue name prefix for vision tasks. The vision tasks can use this method to get
 * unique dispatch queue names which are consistent with other vision tasks.
 * Dispatch queue names need not be unique, but for easy debugging we ensure that the queue names
 * are unique.
 *
 * @param suffix A suffix that identifies a dispatch queue's functionality.
 *
 * @return A unique dispatch queue name by adding the given suffix and a `UUID` to the pre-defined
 * queue name prefix for vision tasks.
 */
+ (const char *)uniqueDispatchQueueNameWithSuffix:(NSString *)suffix;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#ifdef __cplusplus
#include <array>
#include <cstdint>
#include <mutex>
#include <utility>
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"

#include "mediapipe/tasks/c/core/mp_status.h"

namespace mediapipe::tasks::ios::vision::core {

/**
 * Thread-safe pool of static C function-pointer trampolines for MediaPipe vision C API
 * live-stream callbacks.
 *
 * Because the MediaPipe vision C API live-stream `result_callback` signatures do not accept a
 * `void *user_data` context pointer, each live-stream task instance allocates a slot in this pool
 * on initialization (`AllocateSlot`), passes the slot's generated static trampoline (`GetCallback`)
 * to the C API, and releases the slot on teardown (`FreeSlot`). Each slot holds a `__weak`
 * reference to the Objective-C task instance to avoid retain cycles and safely route C callbacks
 * back to the corresponding task instance.
 */
template <typename TaskType, typename ResultType, typename CallbackFnType>
class VisionCallbackPool {
 public:
  static constexpr size_t kMaxSlots = 64;

  struct Slot {
    __weak TaskType *task = nil;
    void (*handler)(TaskType *task, MpStatus status, const ResultType *result, MpImagePtr image,
                    int64_t timestamp_ms) = nullptr;
  };

  static int AllocateSlot(TaskType *task, void (*handler)(TaskType *, MpStatus, const ResultType *,
                                                          MpImagePtr, int64_t)) {
    if (task == nil || handler == nullptr) {
      return -1;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < kMaxSlots; ++i) {
      if (slots_[i].handler == nullptr && slots_[i].task == nil) {
        slots_[i].task = task;
        slots_[i].handler = handler;
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  static void FreeSlot(int slot) {
    if (slot < 0 || slot >= static_cast<int>(kMaxSlots)) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    slots_[slot].task = nil;
    slots_[slot].handler = nullptr;
  }

  static CallbackFnType GetCallback(int slot) {
    if (slot < 0 || slot >= static_cast<int>(kMaxSlots)) {
      return nullptr;
    }
    return trampolines_[slot];
  }

 private:
  static void Dispatch(size_t slot_index, MpStatus status, const ResultType *result,
                       MpImagePtr image, int64_t timestamp_ms) {
    @autoreleasepool {
      TaskType *task = nil;
      void (*handler)(TaskType *, MpStatus, const ResultType *, MpImagePtr, int64_t) = nullptr;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot_index < kMaxSlots) {
          task = slots_[slot_index].task;
          handler = slots_[slot_index].handler;
        }
      }
      if (task && handler) {
        handler(task, status, result, image, timestamp_ms);
      }
    }
  }

  template <size_t N>
  struct Trampoline {
    static void Callback(MpStatus status, const ResultType *result, MpImagePtr image,
                         int64_t timestamp_ms) {
      Dispatch(N, status, result, image, timestamp_ms);
    }
  };

  template <size_t... Is>
  static constexpr auto MakeTrampolines(std::index_sequence<Is...>) {
    return std::array<CallbackFnType, sizeof...(Is)>{(&Trampoline<Is>::Callback)...};
  }

  static inline std::mutex mutex_;
  static inline std::array<Slot, kMaxSlots> slots_;
  static inline const auto trampolines_ = MakeTrampolines(std::make_index_sequence<kMaxSlots>{});
};

}  // namespace mediapipe::tasks::ios::vision::core
#endif
