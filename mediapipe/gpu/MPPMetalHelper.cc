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

#import "mediapipe/gpu/MPPMetalHelper.h"

#import "GTMDefines.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/port/ret_check.h"
#import "mediapipe/gpu/gpu_buffer.h"
#import "mediapipe/gpu/gpu_service.h"
#import "mediapipe/gpu/graph_support.h"
#import "mediapipe/gpu/metal_shared_resources.h"

@interface MPPMetalHelper () {
  mediapipe::GpuResources* _gpuResources;
}
@end

namespace mediapipe {

// Using a C++ class so it can be declared as a friend of
// LegacyCalculatorSupport.
class MetalHelperLegacySupport {
 public:
  static CalculatorContract* GetCalculatorContract() {
    return LegacyCalculatorSupport::Scoped<CalculatorContract>::current();
  }

  static CalculatorContext* GetCalculatorContext() {
    return LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  }
};

}  // namespace mediapipe

@implementation MPPMetalHelper

- (instancetype)initWithGpuResources:(mediapipe::GpuResources*)gpuResources {
  self = [super init];
  if (self) {
    _gpuResources = gpuResources;
  }
  return self;
}

- (instancetype)initWithGpuSharedData:(mediapipe::GpuSharedData*)gpuShared {
  return [self initWithGpuResources:gpuShared->gpu_resources.get()];
}

- (instancetype)initWithCalculatorContext:(mediapipe::CalculatorContext*)cc {
  if (!cc) return nil;
  return [self
      initWithGpuResources:&cc->Service(mediapipe::kGpuService).GetObject()];
}

+ (absl::Status)updateContract:(mediapipe::CalculatorContract*)cc {
  cc->UseService(mediapipe::kGpuService);
  // Allow the legacy side packet to be provided, too, for backwards
  // compatibility with existing graphs. It will just be ignored.
  auto& input_side_packets = cc->InputSidePackets();
  auto id = input_side_packets.GetId(mediapipe::kGpuSharedTagName, 0);
  if (id.IsValid()) {
    input_side_packets.Get(id).Set<mediapipe::GpuSharedData*>();
  }
  return absl::OkStatus();
}

// Legacy support.
- (instancetype)initWithSidePackets:
    (const mediapipe::PacketSet&)inputSidePackets {
  auto cc = mediapipe::MetalHelperLegacySupport::GetCalculatorContext();
  if (cc) {
    ABSL_CHECK_EQ(&inputSidePackets, &cc->InputSidePackets());
    return [self initWithCalculatorContext:cc];
  }

  // TODO: remove when we can.
  ABSL_LOG(WARNING)
      << "CalculatorContext not available. If this calculator uses "
         "CalculatorBase, call initWithCalculatorContext instead.";
  mediapipe::GpuSharedData* gpu_shared =
      inputSidePackets.Tag(mediapipe::kGpuSharedTagName)
          .Get<mediapipe::GpuSharedData*>();

  return [self initWithGpuResources:gpu_shared->gpu_resources.get()];
}

// Legacy support.
+ (absl::Status)setupInputSidePackets:
    (mediapipe::PacketTypeSet*)inputSidePackets {
  auto cc = mediapipe::MetalHelperLegacySupport::GetCalculatorContract();
  if (cc) {
    ABSL_CHECK_EQ(inputSidePackets, &cc->InputSidePackets());
    return [self updateContract:cc];
  }

  // TODO: remove when we can.
  ABSL_LOG(WARNING)
      << "CalculatorContract not available. If you're calling this "
         "from a GetContract method, call updateContract instead.";
  auto id = inputSidePackets->GetId(mediapipe::kGpuSharedTagName, 0);
  RET_CHECK(id.IsValid()) << "A " << mediapipe::kGpuSharedTagName
                          << " input side packet is required here.";
  inputSidePackets->Get(id).Set<mediapipe::GpuSharedData*>();
  return absl::OkStatus();
}

- (id<MTLDevice>)mtlDevice {
  return _gpuResources->metal_shared().resources().mtlDevice;
}

- (id<MTLCommandQueue>)mtlCommandQueue {
  return _gpuResources->metal_shared().resources().mtlCommandQueue;
}

- (CVMetalTextureCacheRef)mtlTextureCache {
  return _gpuResources->metal_shared().resources().mtlTextureCache;
}

- (id<MTLCommandBuffer>)commandBuffer {
  return
      [_gpuResources->metal_shared().resources().mtlCommandQueue commandBuffer];
}

- (CVMetalTextureRef)copyCVMetalTextureWithGpuBuffer:
                         (const mediapipe::GpuBuffer&)gpuBuffer
                                               plane:(size_t)plane {
  CVPixelBufferRef pixel_buffer = mediapipe::GetCVPixelBufferRef(gpuBuffer);
  OSType pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);

  MTLPixelFormat metalPixelFormat = MTLPixelFormatInvalid;
  int width = gpuBuffer.width();
  int height = gpuBuffer.height();

  switch (pixel_format) {
    case kCVPixelFormatType_32BGRA:
      NSCAssert(plane == 0, @"Invalid plane number");
      metalPixelFormat = MTLPixelFormatBGRA8Unorm;
      break;
    case kCVPixelFormatType_64RGBAHalf:
      NSCAssert(plane == 0, @"Invalid plane number");
      metalPixelFormat = MTLPixelFormatRGBA16Float;
      break;
    case kCVPixelFormatType_OneComponent8:
      NSCAssert(plane == 0, @"Invalid plane number");
      metalPixelFormat = MTLPixelFormatR8Uint;
      break;
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
      if (plane == 0) {
        metalPixelFormat = MTLPixelFormatR8Unorm;
      } else if (plane == 1) {
        metalPixelFormat = MTLPixelFormatRG8Unorm;
      } else {
        NSCAssert(NO, @"Invalid plane number");
      }
      width = CVPixelBufferGetWidthOfPlane(pixel_buffer, plane);
      height = CVPixelBufferGetHeightOfPlane(pixel_buffer, plane);
      break;
    case kCVPixelFormatType_TwoComponent16Half:
      metalPixelFormat = MTLPixelFormatRG16Float;
      NSCAssert(plane == 0, @"Invalid plane number");
      break;
    case kCVPixelFormatType_OneComponent32Float:
      metalPixelFormat = MTLPixelFormatR32Float;
      NSCAssert(plane == 0, @"Invalid plane number");
      break;
    default:
      NSCAssert(NO, @"Invalid pixel buffer format");
      break;
  }

  CVMetalTextureRef texture;
  CVReturn err = CVMetalTextureCacheCreateTextureFromImage(
      NULL, _gpuResources->metal_shared().resources().mtlTextureCache,
      mediapipe::GetCVPixelBufferRef(gpuBuffer), NULL, metalPixelFormat, width,
      height, plane, &texture);
  ABSL_CHECK_EQ(err, kCVReturnSuccess);
  return texture;
}

- (CVMetalTextureRef)copyCVMetalTextureWithGpuBuffer:
    (const mediapipe::GpuBuffer&)gpuBuffer {
  return [self copyCVMetalTextureWithGpuBuffer:gpuBuffer plane:0];
}

- (id<MTLTexture>)metalTextureWithGpuBuffer:
    (const mediapipe::GpuBuffer&)gpuBuffer {
  return [self metalTextureWithGpuBuffer:gpuBuffer plane:0];
}

- (id<MTLTexture>)metalTextureWithGpuBuffer:
                      (const mediapipe::GpuBuffer&)gpuBuffer
                                      plane:(size_t)plane {
  CFHolder<CVMetalTextureRef> cvTexture;
  cvTexture.adopt([self copyCVMetalTextureWithGpuBuffer:gpuBuffer plane:plane]);
  return CVMetalTextureGetTexture(*cvTexture);
}

- (mediapipe::GpuBuffer)mediapipeGpuBufferWithWidth:(int)width
                                             height:(int)height {
  auto gpu_buffer = _gpuResources->gpu_buffer_pool().GetBuffer(width, height);
  ABSL_CHECK_OK(gpu_buffer);
  return *gpu_buffer;
}

- (mediapipe::GpuBuffer)mediapipeGpuBufferWithWidth:(int)width
                                             height:(int)height
                                             format:(mediapipe::GpuBufferFormat)
                                                        format {
  auto gpu_buffer =
      _gpuResources->gpu_buffer_pool().GetBuffer(width, height, format);
  ABSL_CHECK_OK(gpu_buffer);
  return *gpu_buffer;
}

- (id<MTLLibrary>)newLibraryWithResourceName:(NSString*)name
                                       error:(NSError* _Nullable*)error {
  return [_gpuResources->metal_shared().resources().mtlDevice
      newLibraryWithFile:[[NSBundle bundleForClass:[self class]]
                             pathForResource:name
                                      ofType:@"metallib"]
                   error:error];
}

@end
