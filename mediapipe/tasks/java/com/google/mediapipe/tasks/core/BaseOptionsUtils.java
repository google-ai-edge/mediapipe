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

import static java.nio.charset.StandardCharsets.UTF_8;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager.NameNotFoundException;
import android.os.ParcelFileDescriptor;
import com.google.common.io.ByteStreams;
import com.google.mediapipe.calculator.proto.InferenceCalculatorProto;
import com.google.mediapipe.tasks.core.proto.AccelerationProto;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.core.proto.ExternalFileProto;
import com.google.mediapipe.tasks.core.proto.ExternalFileProto.ExternalFile;
import com.google.protobuf.ByteString;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

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


  private static final byte[] litertlmMagicBytes = "LITERTLM".getBytes(UTF_8);

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
            modelBuffer ->
                externalFileBuilder.mergeFrom(createExternalFileFromBuffer(modelBuffer)));
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
      case NPU:
        accelerationBuilder.setLitert(
            InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt
                .getDefaultInstance());
        options
            .delegateOptions()
            .ifPresent(
                delegateOptions ->
                    setDelegateOptions(
                        accelerationBuilder,
                        (BaseOptions.DelegateOptions.NpuOptions) delegateOptions));
        break;
      case LITERT:
        accelerationBuilder.setLitert(
            InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt
                .getDefaultInstance());
        options
            .delegateOptions()
            .ifPresent(
                delegateOptions ->
                    setDelegateOptions(
                        accelerationBuilder,
                        (BaseOptions.DelegateOptions.LiteRtOptions) delegateOptions));
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
      BaseOptions.DelegateOptions.NpuOptions options) {
    accelerationBuilder.setLitert(
        InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.newBuilder()
            .setNpu(
                InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Npu.newBuilder()
                    .setDispatchLibraryPath(options.dispatchLibraryDirectory())
                    .setCompilerPluginLibraryPath(options.compilerPluginLibraryDirectory()))
            .build());
  }

  private static void setDelegateOptions(
      AccelerationProto.Acceleration.Builder accelerationBuilder,
      BaseOptions.DelegateOptions.LiteRtOptions options) {
    InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Builder litertBuilder =
        InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.newBuilder();
    options
        .cpuOptions()
        .ifPresent(
            cpuOptions -> {
              InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Cpu.Builder
                  cpuBuilder =
                      InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Cpu
                          .newBuilder();
              litertBuilder.setCpu(cpuBuilder.build());
            });
    options
        .gpuOptions()
        .ifPresent(
            gpuOptions -> {
              InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Gpu.Builder
                  gpuBuilder =
                      InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Gpu
                          .newBuilder();
              InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Gpu.CacheOptions
                      .Builder cacheOptionsBuilder =
                  InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Gpu
                      .CacheOptions.newBuilder();
              gpuOptions.cachedKernelPath().ifPresent(cacheOptionsBuilder::setSerializationDir);
              gpuOptions.modelToken().ifPresent(cacheOptionsBuilder::setModelCacheKey);
              if (gpuOptions.cachedKernelPath().isPresent()
                  && gpuOptions.modelToken().isPresent()) {
                cacheOptionsBuilder.setSerializeProgramCache(true);
              }
              gpuBuilder.setCacheOptions(cacheOptionsBuilder);
              litertBuilder.setGpu(gpuBuilder.build());
            });
    options
        .npuOptions()
        .ifPresent(
            npuOptions -> {
              litertBuilder.setNpu(
                  InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt.Npu
                      .newBuilder()
                      .setDispatchLibraryPath(npuOptions.dispatchLibraryDirectory())
                      .setCompilerPluginLibraryPath(npuOptions.compilerPluginLibraryDirectory())
                      .build());
            });
    accelerationBuilder.setLitert(litertBuilder.build());
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

  public static ExternalFile createExternalFileFromBuffer(ByteBuffer modelBuffer) {
    ExternalFile.Builder externalFileBuilder = ExternalFile.newBuilder();
    if (modelBuffer.isDirect()) {
      externalFileBuilder.setFilePointerMeta(
          ExternalFileProto.FilePointerMeta.newBuilder()
              .setPointer(nativeGetDirectBufferAddress(modelBuffer) + modelBuffer.position())
              .setLength(modelBuffer.remaining())
              .build());
    } else {
      ByteBuffer duplicateBuffer = modelBuffer.duplicate();
      duplicateBuffer.rewind();
      externalFileBuilder.setFileContent(ByteString.copyFrom(duplicateBuffer));
    }
    return externalFileBuilder.build();
  }

  public static boolean isLiteRtLmModel(Context context, BaseOptions baseOptions) {
    if (baseOptions.modelAssetPath().isPresent()) {
      String path = baseOptions.modelAssetPath().get();
      try (InputStream is = context.getAssets().open(path)) {
        return checkLiteRtLmMagicBytes(is);
      } catch (IOException e) {
        try (InputStream is = new FileInputStream(new File(path))) {
          return checkLiteRtLmMagicBytes(is);
        } catch (IOException ex) {
          // Safe to ignore: Treat as non-LiteRT-LM model.
        }
      }
    } else if (baseOptions.modelAssetFileDescriptor().isPresent()) {
      int fd = baseOptions.modelAssetFileDescriptor().get();
      try {
        // Duplicate the file descriptor, so we can reopen it later when the model is loaded.
        ParcelFileDescriptor pfd = ParcelFileDescriptor.fromFd(fd).dup();
        if (pfd != null) {
          try (InputStream is = new ParcelFileDescriptor.AutoCloseInputStream(pfd)) {
            return checkLiteRtLmMagicBytes(is);
          }
        }
      } catch (IOException e) {
        // Ignore
      }
    } else if (baseOptions.modelAssetBuffer().isPresent()) {
      ByteBuffer buffer = baseOptions.modelAssetBuffer().get().duplicate();
      if (buffer.remaining() >= litertlmMagicBytes.length) {
        byte[] bytes = new byte[litertlmMagicBytes.length];
        buffer.get(bytes);
        return Arrays.equals(bytes, litertlmMagicBytes);
      }
    }
    return false;
  }

  private static boolean checkLiteRtLmMagicBytes(InputStream is) throws IOException {
    byte[] bytes = new byte[litertlmMagicBytes.length];
    int read = ByteStreams.read(is, bytes, 0, bytes.length);
    return read == bytes.length && Arrays.equals(bytes, litertlmMagicBytes);
  }

  private static native long nativeGetDirectBufferAddress(ByteBuffer buffer);
}
