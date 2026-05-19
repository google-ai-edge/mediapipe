/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.core;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager.NameNotFoundException;
import com.google.mediapipe.calculator.proto.InferenceCalculatorProto;
import com.google.mediapipe.tasks.core.proto.AccelerationProto;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.core.proto.ExternalFileProto;
import com.google.protobuf.ByteString;

/** Utility for {@link BaseOptions}. */
public final class BaseOptionsUtils {

  // C enum values for HostEnvironment.
  public static final int HOST_ENVIRONMENT_UNKNOWN = 0;
  public static final int HOST_ENVIRONMENT_ANDROID = 1;
  public static final int HOST_ENVIRONMENT_IOS = 2;
  public static final int HOST_ENVIRONMENT_PYTHON = 3;
  public static final int HOST_ENVIRONMENT_WEB = 4;

  // C enum values for HostSystem.
  public static final int HOST_SYSTEM_UNKNOWN = 0;
  public static final int HOST_SYSTEM_LINUX = 1;
  public static final int HOST_SYSTEM_MAC = 2;
  public static final int HOST_SYSTEM_WINDOWS = 3;
  public static final int HOST_SYSTEM_IOS = 4;
  public static final int HOST_SYSTEM_ANDROID = 5;


  private BaseOptionsUtils() {}

  /** Returns the app id of the host environment, e.g., Android package name. */
  public static String getAppId(Context context) {
    return context.getPackageName();
  }

  /** Returns the app version of the host environment, e.g., Android version code. */
  public static String getAppVersion(Context context) {
    try {
      PackageInfo packageInfo =
          context.getPackageManager().getPackageInfo(context.getPackageName(), 0);
      if (packageInfo != null) {
        return String.valueOf(packageInfo.versionCode);
      }
    } catch (NameNotFoundException e) {
      // Ignore exception and return fallback.
    }
    return "<not found>";
  }

  /**
   * Converts a {@link BaseOptions} instance to a {@link BaseOptionsProto.BaseOptions} protobuf
   * message.
   */
  public static BaseOptionsProto.BaseOptions convertBaseOptionsToProto(BaseOptions options) {
    ExternalFileProto.ExternalFile.Builder externalFileBuilder =
        ExternalFileProto.ExternalFile.newBuilder();
    options.modelAssetPath().ifPresent(externalFileBuilder::setFileName);
    options
        .modelAssetFileDescriptor()
        .ifPresent(
            fd ->
                externalFileBuilder.setFileDescriptorMeta(
                    ExternalFileProto.FileDescriptorMeta.newBuilder().setFd(fd).build()));
    options
        .modelAssetBuffer()
        .ifPresent(
            modelBuffer -> {
              modelBuffer.rewind();
              externalFileBuilder.setFileContent(ByteString.copyFrom(modelBuffer));
            });
    AccelerationProto.Acceleration.Builder accelerationBuilder =
        AccelerationProto.Acceleration.newBuilder();
    switch (options.delegate()) {
      case CPU:
        accelerationBuilder.setTflite(
            InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.TfLite
                .getDefaultInstance());
        options
            .delegateOptions()
            .ifPresent(
                delegateOptions ->
                    setDelegateOptions(
                        accelerationBuilder,
                        (BaseOptions.DelegateOptions.CpuOptions) delegateOptions));
        break;
      case GPU:
        accelerationBuilder.setGpu(
            InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.Gpu.newBuilder()
                .setUseAdvancedGpuApi(true)
                .build());
        options
            .delegateOptions()
            .ifPresent(
                delegateOptions ->
                    setDelegateOptions(
                        accelerationBuilder,
                        (BaseOptions.DelegateOptions.GpuOptions) delegateOptions));
        break;
    }

    return BaseOptionsProto.BaseOptions.newBuilder()
        .setModelAsset(externalFileBuilder.build())
        .setAcceleration(accelerationBuilder.build())
        .build();
  }

  private static void setDelegateOptions(
      AccelerationProto.Acceleration.Builder accelerationBuilder,
      BaseOptions.DelegateOptions.CpuOptions options) {
    accelerationBuilder.setTflite(
        InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.TfLite.getDefaultInstance());
  }

  private static void setDelegateOptions(
      AccelerationProto.Acceleration.Builder accelerationBuilder,
      BaseOptions.DelegateOptions.GpuOptions options) {
    InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.Gpu.Builder gpuBuilder =
        InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.Gpu.newBuilder()
            .setUseAdvancedGpuApi(true);
    options.cachedKernelPath().ifPresent(gpuBuilder::setCachedKernelPath);
    options.serializedModelDir().ifPresent(gpuBuilder::setSerializedModelDir);
    options.modelToken().ifPresent(gpuBuilder::setModelToken);
    accelerationBuilder.setGpu(gpuBuilder.build());
  }
}
