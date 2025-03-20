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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_DIFFUSER_DIFFUSER_GPU_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_DIFFUSER_DIFFUSER_GPU_H_

#include <limits.h>
#include <stdint.h>

#ifndef DG_EXPORT
#define DG_EXPORT __attribute__((visibility("default")))
#endif  // DG_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

enum DiffuserModelType {
  kDiffuserModelTypeSd1,
  kDiffuserModelTypeGldm,
  kDiffuserModelTypeDistilledGldm,
  kDiffuserModelTypeSd2Base,
  kDiffuserModelTypeTigo,
  kDiffuserModelTypeTigoUfo,
};

enum DiffuserPriorityHint {
  kDiffuserPriorityHintHigh,
  kDiffuserPriorityHintNormal,
  kDiffuserPriorityHintLow,
};

enum DiffuserPerformanceHint {
  kDiffuserPerformanceHintHigh,
  kDiffuserPerformanceHintNormal,
  kDiffuserPerformanceHintLow,
};

typedef struct {
  DiffuserPriorityHint priority_hint;
  DiffuserPerformanceHint performance_hint;
} DiffuserEnvironmentOptions;

typedef struct {
  DiffuserModelType model_type;
  char model_dir[PATH_MAX];
  char lora_dir[PATH_MAX];
  const void* lora_weights_layer_mapping;
  int lora_rank;
  int seed;
  int image_width;
  int image_height;
  int run_unet_with_plugins;
  int run_unet_with_masked_image;
  DiffuserEnvironmentOptions env_options;
} DiffuserConfig;

typedef struct {
  void* diffuser;
} DiffuserContext;

typedef struct {
  int shape[4];
  const float* data;
} DiffuserPluginTensor;

DG_EXPORT DiffuserContext* DiffuserCreate(const DiffuserConfig*);  // NOLINT
DG_EXPORT int DiffuserReset(DiffuserContext*,                      // NOLINT
                            const char*, int, int, float, const void*);
DG_EXPORT int DiffuserIterate(DiffuserContext*, int, int);  // NOLINT
DG_EXPORT int DiffuserDecode(DiffuserContext*, uint8_t*);   // NOLINT
DG_EXPORT void DiffuserDelete(DiffuserContext*);            // NOLINT

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_GENERATOR_IMAGE_GENERATOR_DIFFUSER_DIFFUSER_GPU_H_
