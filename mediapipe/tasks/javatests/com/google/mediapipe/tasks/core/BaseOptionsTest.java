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

package com.google.mediapipe.tasks.core;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.tasks.core.proto.AccelerationProto;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Test for {@link BaseOptions} */
@RunWith(Suite.class)
@SuiteClasses({BaseOptionsTest.General.class, BaseOptionsTest.ConvertProtoTest.class})
public class BaseOptionsTest {

  static final String MODEL_ASSET_PATH = "dummy_model.tflite";
  static final String SERIALIZED_MODEL_DIR = "dummy_serialized_model_dir";
  static final String MODEL_TOKEN = "dummy_model_token";
  static final String CACHED_KERNEL_PATH = "dummy_cached_kernel_path";

  @RunWith(AndroidJUnit4.class)
  public static final class General extends BaseOptionsTest {
    @Test
    public void succeedsWithDefaultOptions() throws Exception {
      BaseOptions options = BaseOptions.builder().setModelAssetPath(MODEL_ASSET_PATH).build();
      assertThat(options.modelAssetPath().isPresent()).isTrue();
      assertThat(options.modelAssetPath().get()).isEqualTo(MODEL_ASSET_PATH);
      assertThat(options.delegate()).isEqualTo(Delegate.CPU);
    }

    @Test
    public void succeedsWithGpuOptions() throws Exception {
      BaseOptions options =
          BaseOptions.builder()
              .setModelAssetPath(MODEL_ASSET_PATH)
              .setDelegate(Delegate.GPU)
              .setDelegateOptions(
                  BaseOptions.DelegateOptions.GpuOptions.builder()
                      .setSerializedModelDir(SERIALIZED_MODEL_DIR)
                      .setModelToken(MODEL_TOKEN)
                      .setCachedKernelPath(CACHED_KERNEL_PATH)
                      .build())
              .build();
      assertThat(
              ((BaseOptions.DelegateOptions.GpuOptions) options.delegateOptions().get())
                  .serializedModelDir()
                  .get())
          .isEqualTo(SERIALIZED_MODEL_DIR);
      assertThat(
              ((BaseOptions.DelegateOptions.GpuOptions) options.delegateOptions().get())
                  .modelToken()
                  .get())
          .isEqualTo(MODEL_TOKEN);
      assertThat(
              ((BaseOptions.DelegateOptions.GpuOptions) options.delegateOptions().get())
                  .cachedKernelPath()
                  .get())
          .isEqualTo(CACHED_KERNEL_PATH);
    }

    @Test
    public void failsWithInvalidDelegateOptions() throws Exception {
      IllegalArgumentException exception =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  BaseOptions.builder()
                      .setModelAssetPath(MODEL_ASSET_PATH)
                      .setDelegate(Delegate.CPU)
                      .setDelegateOptions(
                          BaseOptions.DelegateOptions.GpuOptions.builder()
                              .setSerializedModelDir(SERIALIZED_MODEL_DIR)
                              .setModelToken(MODEL_TOKEN)
                              .build())
                      .build());
      assertThat(exception)
          .hasMessageThat()
          .contains("Specified Delegate type does not match the provided delegate options.");
    }
  }

  /** A mock TaskOptions class providing access to convertBaseOptionsToProto. */
  public static class MockTaskOptions extends TaskOptions {

    public MockTaskOptions(BaseOptions baseOptions) {
      baseOptionsProto = convertBaseOptionsToProto(baseOptions);
    }

    public BaseOptionsProto.BaseOptions getBaseOptionsProto() {
      return baseOptionsProto;
    }

    private BaseOptionsProto.BaseOptions baseOptionsProto;

    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      return CalculatorOptions.newBuilder().build();
    }
  }

  /** Test for converting {@link BaseOptions} to {@link BaseOptionsProto} */
  @RunWith(AndroidJUnit4.class)
  public static final class ConvertProtoTest extends BaseOptionsTest {
    @Test
    public void succeedsWithDefaultOptions() throws Exception {
      BaseOptions options =
          BaseOptions.builder()
              .setModelAssetPath(MODEL_ASSET_PATH)
              .setDelegate(Delegate.CPU)
              .setDelegateOptions(BaseOptions.DelegateOptions.CpuOptions.builder().build())
              .build();
      MockTaskOptions taskOptions = new MockTaskOptions(options);
      AccelerationProto.Acceleration acceleration =
          taskOptions.getBaseOptionsProto().getAcceleration();
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
      MockTaskOptions taskOptions = new MockTaskOptions(options);
      AccelerationProto.Acceleration acceleration =
          taskOptions.getBaseOptionsProto().getAcceleration();
      assertThat(acceleration.hasTflite()).isFalse();
      assertThat(acceleration.hasGpu()).isTrue();
      assertThat(acceleration.getGpu().getUseAdvancedGpuApi()).isTrue();
      assertThat(acceleration.getGpu().hasCachedKernelPath()).isFalse();
      assertThat(acceleration.getGpu().getModelToken()).isEqualTo(MODEL_TOKEN);
      assertThat(acceleration.getGpu().getSerializedModelDir()).isEqualTo(SERIALIZED_MODEL_DIR);
    }
  }
}
