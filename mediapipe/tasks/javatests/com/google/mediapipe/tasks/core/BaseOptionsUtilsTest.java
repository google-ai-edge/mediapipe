// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.core;

import static com.google.common.truth.Truth.assertThat;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.tasks.core.proto.AccelerationProto;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for {@link BaseOptionsUtils} */
@RunWith(AndroidJUnit4.class)
public class BaseOptionsUtilsTest {

  static final String MODEL_ASSET_PATH = "dummy_model.tflite";
  static final String SERIALIZED_MODEL_DIR = "dummy_serialized_model_dir";
  static final String MODEL_TOKEN = "dummy_model_token";
  static final String CACHED_KERNEL_PATH = "dummy_cached_kernel_path";
  static final String DISPATCH_LIBRARY_PATH = "dummy_dispatch_library_path";
  static final String COMPILER_PLUGIN_LIBRARY_PATH = "dummy_compiler_plugin_library_path";

  @Test
  public void succeedsWithDefaultOptions() throws Exception {
    BaseOptions options =
        BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET_PATH)
            .setDelegate(Delegate.CPU)
            .setDelegateOptions(BaseOptions.DelegateOptions.CpuOptions.builder().build())
            .build();
    BaseOptionsProto.BaseOptions baseOptionsProto =
        BaseOptionsUtils.convertBaseOptionsToProto(options);
    AccelerationProto.Acceleration acceleration = baseOptionsProto.getAcceleration();
    assertThat(acceleration.hasTflite()).isTrue();
  }

  @Test
  public void succeedsWithGpuOptions() throws Exception {
    BaseOptions options =
        BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET_PATH)
            .setDelegate(Delegate.GPU)
            .setDelegateOptions(
                BaseOptions.DelegateOptions.GpuOptions.builder()
                    .setModelToken(MODEL_TOKEN)
                    .setSerializedModelDir(SERIALIZED_MODEL_DIR)
                    .build())
            .build();
    BaseOptionsProto.BaseOptions baseOptionsProto =
        BaseOptionsUtils.convertBaseOptionsToProto(options);
    AccelerationProto.Acceleration acceleration = baseOptionsProto.getAcceleration();
    assertThat(acceleration.hasTflite()).isFalse();
    assertThat(acceleration.hasGpu()).isTrue();
    assertThat(acceleration.getGpu().getUseAdvancedGpuApi()).isTrue();
    assertThat(acceleration.getGpu().hasCachedKernelPath()).isFalse();
    assertThat(acceleration.getGpu().getModelToken()).isEqualTo(MODEL_TOKEN);
    assertThat(acceleration.getGpu().getSerializedModelDir()).isEqualTo(SERIALIZED_MODEL_DIR);
  }

  @Test
  public void succeedsWithNpuOptions() throws Exception {
    BaseOptions options =
        BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET_PATH)
            .setDelegate(Delegate.NPU)
            .setDelegateOptions(
                BaseOptions.DelegateOptions.NpuOptions.builder()
                    .setDispatchLibraryDirectory(DISPATCH_LIBRARY_PATH)
                    .setCompilerPluginLibraryDirectory(COMPILER_PLUGIN_LIBRARY_PATH)
                    .build())
            .build();
    BaseOptionsProto.BaseOptions baseOptionsProto =
        BaseOptionsUtils.convertBaseOptionsToProto(options);
    AccelerationProto.Acceleration acceleration = baseOptionsProto.getAcceleration();
    assertThat(acceleration.hasTflite()).isFalse();
    assertThat(acceleration.hasLitert()).isTrue();
    assertThat(acceleration.getLitert().hasNpu()).isTrue();
    assertThat(acceleration.getLitert().getNpu().getDispatchLibraryPath())
        .isEqualTo(DISPATCH_LIBRARY_PATH);
    assertThat(acceleration.getLitert().getNpu().getCompilerPluginLibraryPath())
        .isEqualTo(COMPILER_PLUGIN_LIBRARY_PATH);
  }
}
