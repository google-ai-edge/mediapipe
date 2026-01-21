// Copyright 2025 The MediaPipe Authors.
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

package com.google.mediapipe.framework;

import java.util.Map;

/** Manages MediaPipe resources. */
public final class ResourcesWithMapping {

  private final Map<String, String> resourcesMapping;

  public ResourcesWithMapping() {
    this(null);
  }

  public ResourcesWithMapping(Map<String, String> resourcesMapping) {
    this.resourcesMapping = resourcesMapping;
  }

  public Map<String, String> getResourcesMapping() {
    return resourcesMapping;
  }
}
