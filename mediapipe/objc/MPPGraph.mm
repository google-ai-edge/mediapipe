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

#import "mediapipe/objc/MPPGraph.h"

#import <AVFoundation/AVFoundation.h>
#import <Accelerate/Accelerate.h>

#include <atomic>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/gpu/MPPGraphGPUData.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/objc/util.h"

#import "mediapipe/objc/NSError+util_status.h"
#import "GTMDefines.h"

@implementation MPPGraph {
  // Graph is wrapped in a unique_ptr because it was generating 39+KB of unnecessary ObjC runtime
  // information. See https://medium.com/@dmaclach/objective-c-encoding-and-you-866624cc02de
  // for details.
  std::unique_ptr<mediapipe::CalculatorGraph> _graph;
  /// Input side packets that will be added to the graph when it is started.
  std::map<std::string, mediapipe::Packet> _inputSidePackets;
  /// Packet headers that will be added to the graph when it is started.
  std::map<std::string, mediapipe::Packet> _streamHeaders;
  /// Service packets to be added to the graph when it is started.
  std::map<const mediapipe::GraphServiceBase*, mediapipe::Packet> _servicePackets;

  /// Number of frames currently being processed by the graph.
  std::atomic<int32_t> _framesInFlight;
  /// Used as a sequential timestamp for MediaPipe.
  mediapipe::Timestamp _frameTimestamp;
  int64 _frameNumber;

  // Graph config modified to expose requested output streams.
  mediapipe::CalculatorGraphConfig _config;

  // Tracks whether the graph has been started and is currently running.
  BOOL _started;
}

- (instancetype)initWithGraphConfig:(const mediapipe::CalculatorGraphConfig&)config {
  self = [super init];
  if (self) {
    // Turn on Cocoa multithreading, since MediaPipe uses threads.
    // Not needed on iOS, but we may want to have OS X clients in the future.
    [[[NSThread alloc] init] start];
    _graph = absl::make_unique<mediapipe::CalculatorGraph>();
    _config = config;
  }
  return self;
}

- (mediapipe::ProfilingContext*)getProfiler {
  return _graph->profiler();
}

- (mediapipe::CalculatorGraph::GraphInputStreamAddMode)packetAddMode {
  return _graph->GetGraphInputStreamAddMode();
}

- (void)setPacketAddMode:(mediapipe::CalculatorGraph::GraphInputStreamAddMode)mode {
  _graph->SetGraphInputStreamAddMode(mode);
}

- (void)addFrameOutputStream:(const std::string&)outputStreamName
            outputPacketType:(MPPPacketType)packetType {
  std::string callbackInputName;
  mediapipe::tool::AddCallbackCalculator(outputStreamName, &_config, &callbackInputName,
                                       /*use_std_function=*/true);
  // No matter what ownership qualifiers are put on the pointer, NewPermanentCallback will
  // still end up with a strong pointer to MPPGraph*. That is why we use void* instead.
  void* wrapperVoid = (__bridge void*)self;
  _inputSidePackets[callbackInputName] =
      mediapipe::MakePacket<std::function<void(const mediapipe::Packet&)>>(
          [wrapperVoid, outputStreamName, packetType](const mediapipe::Packet& packet) {
            CallFrameDelegate(wrapperVoid, outputStreamName, packetType, packet);
          });
}

- (NSString *)description {
  return [NSString stringWithFormat:@"<%@: %p; framesInFlight = %d>", [self class], self,
                                    _framesInFlight.load(std::memory_order_relaxed)];
}

/// This is the function that gets called by the CallbackCalculator that
/// receives the graph's output.
void CallFrameDelegate(void* wrapperVoid, const std::string& streamName,
                       MPPPacketType packetType, const mediapipe::Packet& packet) {
  MPPGraph* wrapper = (__bridge MPPGraph*)wrapperVoid;
  @autoreleasepool {
    if (packetType == MPPPacketTypeRaw) {
      [wrapper.delegate mediapipeGraph:wrapper
                     didOutputPacket:packet
                          fromStream:streamName];
    } else if (packetType == MPPPacketTypeImageFrame) {
      const auto& frame = packet.Get<mediapipe::ImageFrame>();
      mediapipe::ImageFormat::Format format = frame.Format();

      if (format == mediapipe::ImageFormat::SRGBA ||
          format == mediapipe::ImageFormat::GRAY8) {
        CVPixelBufferRef pixelBuffer;
        // If kCVPixelFormatType_32RGBA does not work, it returns kCVReturnInvalidPixelFormat.
        CVReturn error = CVPixelBufferCreate(
            NULL, frame.Width(), frame.Height(), kCVPixelFormatType_32BGRA,
            GetCVPixelBufferAttributesForGlCompatibility(), &pixelBuffer);
        _GTMDevAssert(error == kCVReturnSuccess, @"CVPixelBufferCreate failed: %d", error);
        error = CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        _GTMDevAssert(error == kCVReturnSuccess, @"CVPixelBufferLockBaseAddress failed: %d", error);

        vImage_Buffer vDestination = vImageForCVPixelBuffer(pixelBuffer);
        // Note: we have to throw away const here, but we should not overwrite
        // the packet data.
        vImage_Buffer vSource = vImageForImageFrame(frame);
        if (format == mediapipe::ImageFormat::SRGBA) {
          // Swap R and B channels.
          const uint8_t permuteMap[4] = {2, 1, 0, 3};
          vImage_Error vError = vImagePermuteChannels_ARGB8888(
              &vSource, &vDestination, permuteMap, kvImageNoFlags);
          _GTMDevAssert(vError == kvImageNoError, @"vImagePermuteChannels failed: %zd", vError);
        } else {
          // Convert grayscale back to BGRA
          vImage_Error vError = vImageGrayToBGRA(&vSource, &vDestination);
          _GTMDevAssert(vError == kvImageNoError, @"vImageGrayToBGRA failed: %zd", vError);
        }

        error = CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        _GTMDevAssert(error == kCVReturnSuccess,
                      @"CVPixelBufferUnlockBaseAddress failed: %d", error);

        if ([wrapper.delegate respondsToSelector:@selector
                              (mediapipeGraph:didOutputPixelBuffer:fromStream:timestamp:)]) {
          [wrapper.delegate mediapipeGraph:wrapper
                    didOutputPixelBuffer:pixelBuffer
                              fromStream:streamName
                               timestamp:packet.Timestamp()];
        } else if ([wrapper.delegate respondsToSelector:@selector
                                     (mediapipeGraph:didOutputPixelBuffer:fromStream:)]) {
          [wrapper.delegate mediapipeGraph:wrapper
                    didOutputPixelBuffer:pixelBuffer
                              fromStream:streamName];
        }
        CVPixelBufferRelease(pixelBuffer);
      } else {
        _GTMDevLog(@"unsupported ImageFormat: %d", format);
      }
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    } else if (packetType == MPPPacketTypePixelBuffer) {
      CVPixelBufferRef pixelBuffer = packet.Get<mediapipe::GpuBuffer>().GetCVPixelBufferRef();
      if ([wrapper.delegate
              respondsToSelector:@selector
              (mediapipeGraph:didOutputPixelBuffer:fromStream:timestamp:)]) {
        [wrapper.delegate mediapipeGraph:wrapper
                  didOutputPixelBuffer:pixelBuffer
                            fromStream:streamName
                             timestamp:packet.Timestamp()];
      } else if ([wrapper.delegate
                     respondsToSelector:@selector
                     (mediapipeGraph:didOutputPixelBuffer:fromStream:)]) {
        [wrapper.delegate mediapipeGraph:wrapper
                  didOutputPixelBuffer:pixelBuffer
                            fromStream:streamName];
      }
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    } else {
      _GTMDevLog(@"unsupported packet type");
    }

    wrapper->_framesInFlight--;
  }
}

- (void)setHeaderPacket:(const mediapipe::Packet&)packet forStream:(const std::string&)streamName {
  _GTMDevAssert(!_started, @"%@ must be called before the graph is started",
                NSStringFromSelector(_cmd));
  _streamHeaders[streamName] = packet;
}

- (void)setSidePacket:(const mediapipe::Packet&)packet named:(const std::string&)name {
  _GTMDevAssert(!_started, @"%@ must be called before the graph is started",
                NSStringFromSelector(_cmd));
  _inputSidePackets[name] = packet;
}

- (void)setServicePacket:(mediapipe::Packet&)packet
              forService:(const mediapipe::GraphServiceBase&)service {
  _GTMDevAssert(!_started, @"%@ must be called before the graph is started",
                NSStringFromSelector(_cmd));
  _servicePackets[&service] = std::move(packet);
}

- (void)addSidePackets:(const std::map<std::string, mediapipe::Packet>&)extraSidePackets {
  _GTMDevAssert(!_started, @"%@ must be called before the graph is started",
                NSStringFromSelector(_cmd));
  _inputSidePackets.insert(extraSidePackets.begin(), extraSidePackets.end());
}

- (BOOL)startWithError:(NSError**)error {
  ::mediapipe::Status status = [self performStart];
  if (!status.ok()) {
    if (error) {
      *error = [NSError gus_errorWithStatus:status];
    }
    return NO;
  }
  _started = YES;
  return YES;
}

- (::mediapipe::Status)performStart {
  ::mediapipe::Status status = _graph->Initialize(_config);
  if (!status.ok()) {
    return status;
  }
  for (const auto& service_packet : _servicePackets) {
    status = _graph->SetServicePacket(*service_packet.first, service_packet.second);
    if (!status.ok()) {
      return status;
    }
  }
  status = _graph->StartRun(_inputSidePackets, _streamHeaders);
  if (!status.ok()) {
    return status;
  }
  return status;
}

- (void)cancel {
  _graph->Cancel();
}

- (BOOL)hasInputStream:(const std::string&)inputName {
  return _graph->HasInputStream(inputName);
}

- (BOOL)closeInputStream:(const std::string&)inputName error:(NSError**)error {
  ::mediapipe::Status status = _graph->CloseInputStream(inputName);
  if (!status.ok() && error) *error = [NSError gus_errorWithStatus:status];
  return status.ok();
}

- (BOOL)closeAllInputStreamsWithError:(NSError**)error {
  ::mediapipe::Status status = _graph->CloseAllInputStreams();
  if (!status.ok() && error) *error = [NSError gus_errorWithStatus:status];
  return status.ok();
}

- (BOOL)waitUntilDoneWithError:(NSError**)error {
  // Since this method blocks with no timeout, it should not be called in the main thread in
  // an app. However, it's fine to allow that in a test.
  // TODO: is this too heavy-handed? Maybe a warning would be fine.
  _GTMDevAssert(![NSThread isMainThread] || (NSClassFromString(@"XCTest")),
                @"waitUntilDoneWithError: should not be called on the main thread");
  ::mediapipe::Status status = _graph->WaitUntilDone();
  _started = NO;
  if (!status.ok() && error) *error = [NSError gus_errorWithStatus:status];
  return status.ok();
}

- (BOOL)waitUntilIdleWithError:(NSError**)error {
  ::mediapipe::Status status = _graph->WaitUntilIdle();
  if (!status.ok() && error) *error = [NSError gus_errorWithStatus:status];
  return status.ok();
}

- (BOOL)movePacket:(mediapipe::Packet&&)packet
        intoStream:(const std::string&)streamName
             error:(NSError**)error {
  ::mediapipe::Status status = _graph->AddPacketToInputStream(streamName, std::move(packet));
  if (!status.ok() && error) *error = [NSError gus_errorWithStatus:status];
  return status.ok();
}

- (BOOL)sendPacket:(const mediapipe::Packet&)packet
        intoStream:(const std::string&)streamName
             error:(NSError**)error {
  ::mediapipe::Status status = _graph->AddPacketToInputStream(streamName, packet);
  if (!status.ok() && error) *error = [NSError gus_errorWithStatus:status];
  return status.ok();
}

- (BOOL)setMaxQueueSize:(int)maxQueueSize
              forStream:(const std::string&)streamName
                  error:(NSError**)error {
  ::mediapipe::Status status = _graph->SetInputStreamMaxQueueSize(streamName, maxQueueSize);
  if (!status.ok() && error) *error = [NSError gus_errorWithStatus:status];
  return status.ok();
}

- (mediapipe::Packet)packetWithPixelBuffer:(CVPixelBufferRef)imageBuffer
                              packetType:(MPPPacketType)packetType {
  mediapipe::Packet packet;
  if (packetType == MPPPacketTypeImageFrame || packetType == MPPPacketTypeImageFrameBGRANoSwap) {
    auto frame = CreateImageFrameForCVPixelBuffer(
        imageBuffer, /* canOverwrite = */ false,
        /* bgrAsRgb = */ packetType == MPPPacketTypeImageFrameBGRANoSwap);
    packet = mediapipe::Adopt(frame.release());
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  } else if (packetType == MPPPacketTypePixelBuffer) {
    packet = mediapipe::MakePacket<mediapipe::GpuBuffer>(imageBuffer);
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  } else {
    _GTMDevLog(@"unsupported packet type: %d", packetType);
  }
  return packet;
}

- (BOOL)sendPixelBuffer:(CVPixelBufferRef)imageBuffer
             intoStream:(const std::string&)inputName
             packetType:(MPPPacketType)packetType
              timestamp:(const mediapipe::Timestamp&)timestamp
         allowOverwrite:(BOOL)allowOverwrite {
  NSError* error;
  bool success = [self sendPixelBuffer:imageBuffer
                            intoStream:inputName
                            packetType:packetType
                             timestamp:timestamp
                        allowOverwrite:allowOverwrite
                                 error:&error];
  if (error) {
    _GTMDevLog(@"failed to send packet: %@", error);
  }
  return success;
}

- (BOOL)sendPixelBuffer:(CVPixelBufferRef)imageBuffer
             intoStream:(const std::string&)inputName
             packetType:(MPPPacketType)packetType
              timestamp:(const mediapipe::Timestamp&)timestamp
         allowOverwrite:(BOOL)allowOverwrite
                  error:(NSError**)error {
  if (_maxFramesInFlight && _framesInFlight >= _maxFramesInFlight) return NO;
  mediapipe::Packet packet = [self packetWithPixelBuffer:imageBuffer packetType:packetType];
  BOOL success;
  if (allowOverwrite) {
    packet = std::move(packet).At(timestamp);
    success = [self movePacket:std::move(packet) intoStream:inputName error:error];
  } else {
    success = [self sendPacket:packet.At(timestamp) intoStream:inputName error:error];
  }
  if (success) _framesInFlight++;
  return success;
}

- (BOOL)sendPixelBuffer:(CVPixelBufferRef)imageBuffer
             intoStream:(const std::string&)inputName
             packetType:(MPPPacketType)packetType
              timestamp:(const mediapipe::Timestamp&)timestamp {
  return [self sendPixelBuffer:imageBuffer
                    intoStream:inputName
                    packetType:packetType
                     timestamp:timestamp
                allowOverwrite:NO];
}

- (BOOL)sendPixelBuffer:(CVPixelBufferRef)imageBuffer
             intoStream:(const std::string&)inputName
             packetType:(MPPPacketType)packetType {
  _GTMDevAssert(_frameTimestamp < mediapipe::Timestamp::Done(),
                @"Trying to send frame after stream is done.");
  if (_frameTimestamp < mediapipe::Timestamp::Min()) {
    _frameTimestamp = mediapipe::Timestamp::Min();
  } else {
    _frameTimestamp++;
  }
  return [self sendPixelBuffer:imageBuffer
                    intoStream:inputName
                    packetType:packetType
                     timestamp:_frameTimestamp];
}

- (void)debugPrintGlInfo {
  std::shared_ptr<mediapipe::GpuResources> gpu_resources = _graph->GetGpuResources();
  if (!gpu_resources) {
    NSLog(@"GPU not set up.");
    return;
  }

  NSString* extensionString;
  (void)gpu_resources->gl_context()->Run([&extensionString]{
    extensionString = [NSString stringWithUTF8String:(char*)glGetString(GL_EXTENSIONS)];
    return ::mediapipe::OkStatus();
  });

  NSArray* extensions = [extensionString componentsSeparatedByCharactersInSet:
                         [NSCharacterSet whitespaceCharacterSet]];
  for (NSString* oneExtension in extensions)
    NSLog(@"%@", oneExtension);
}

@end
