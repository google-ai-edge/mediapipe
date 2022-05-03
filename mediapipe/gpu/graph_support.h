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

// TODO: Update all the reference and delete this forwarding header.
#ifndef MEDIAPIPE_GPU_GRAPH_SUPPORT_H_
#define MEDIAPIPE_GPU_GRAPH_SUPPORT_H_

#include "mediapipe/framework/graph_service.h"

namespace mediapipe {

// Forward declaration to avoid depending on GpuResources here.
class GpuResources;
extern const GraphService<GpuResources> kGpuService;

static constexpr char kGpuSharedTagName[] = "GPU_SHARED";
static constexpr char kGpuSharedSidePacketName[] = "gpu_shared";
static constexpr char kGpuExecutorName[] = "__gpu";

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GRAPH_SUPPORT_H_
