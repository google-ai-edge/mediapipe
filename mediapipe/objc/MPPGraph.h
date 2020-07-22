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

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#ifndef __cplusplus
#error This header can only be included by an Objective-C++ file.
#endif

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/objc/util.h"

@class MPPGraph;

namespace mediapipe {
struct GpuSharedData;
}  // namespace mediapipe

/// A delegate that can receive frames from a MediaPipe graph.
@protocol MPPGraphDelegate <NSObject>

/// Provides the delegate with a new video frame.
@optional
- (void)mediapipeGraph:(MPPGraph *)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string &)streamName;

/// Provides the delegate with a new video frame and time stamp.
@optional
- (void)mediapipeGraph:(MPPGraph *)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string &)streamName
               timestamp:(const mediapipe::Timestamp &)timestamp;

/// Provides the delegate with a raw packet.
@optional
- (void)mediapipeGraph:(MPPGraph *)graph
       didOutputPacket:(const mediapipe::Packet &)packet
            fromStream:(const std::string &)streamName;

@end

/// Chooses the packet type used by MPPGraph to send and receive packets
/// from the graph.
typedef NS_ENUM(int, MPPPacketType) {
  /// Any packet type.
  /// Calls mediapipeGraph:didOutputPacket:fromStream:
  MPPPacketTypeRaw,

  /// CFHolder<CVPixelBufferRef>.
  /// Calls mediapipeGraph:didOutputPixelBuffer:fromStream:
  /// Use this packet type to pass GPU frames to calculators.
  MPPPacketTypePixelBuffer,

  /// ImageFrame.
  /// Calls mediapipeGraph:didOutputPixelBuffer:fromStream:
  MPPPacketTypeImageFrame,

  /// RGBA ImageFrame, but do not swap the channels if the input pixel buffer
  /// is BGRA. This is useful when the graph needs RGBA ImageFrames, but the
  /// calculators do not care about the order of the channels, so BGRA data can
  /// be used as-is.
  /// Calls mediapipeGraph:didOutputPixelBuffer:fromStream:
  MPPPacketTypeImageFrameBGRANoSwap,
};

/// This class is an Objective-C wrapper around a MediaPipe graph object, and
/// helps interface it with iOS technologies such as AVFoundation.
@interface MPPGraph : NSObject

/// The delegate, which receives output frames.
@property(weak) id<MPPGraphDelegate> delegate;

/// If the graph is already processing more than this number of frames, drop any
/// new incoming frames. Used to avoid swamping slower devices when processing
/// cannot keep up with the speed of video input.
/// This works as long as frames are sent or received using these methods:
///  - sendPixelBuffer:intoStream:packetType:[timestamp:]
///  - addFrameOutputStream:outputPacketType:
/// Set to 0 (the default) for no limit.
@property(nonatomic) int maxFramesInFlight;

/// Determines whether adding a packet to an input stream whose queue is full
/// should fail or wait.
@property mediapipe::CalculatorGraph::GraphInputStreamAddMode packetAddMode;

- (instancetype)init NS_UNAVAILABLE;

/// Copies the config and initializes the graph.
/// @param config The configuration describing the graph.
- (instancetype)initWithGraphConfig:(const mediapipe::CalculatorGraphConfig &)config
    NS_DESIGNATED_INITIALIZER;

- (mediapipe::ProfilingContext *)getProfiler;

/// Sets a stream header. If the header was already set, it is overwritten.
/// @param packet The header.
/// @param streamName The name of the stream.
- (void)setHeaderPacket:(const mediapipe::Packet &)packet forStream:(const std::string &)streamName;

/// Sets a side packet. If it was already set, it is overwritten.
/// Must be called before the graph is started.
/// @param packet The packet to be associated with the input side packet.
/// @param name The name of the input side packet.
- (void)setSidePacket:(const mediapipe::Packet &)packet named:(const std::string &)name;

/// Sets a service packet. If it was already set, it is overwritten.
/// Must be called before the graph is started.
/// @param packet The packet to be associated with the service.
/// @param service.
- (void)setServicePacket:(mediapipe::Packet &)packet
              forService:(const mediapipe::GraphServiceBase &)service;

/// Adds input side packets from a map. Any inputs that were already set are
/// left unchanged.
/// Must be called before the graph is started.
/// @param extraInputSidePackets The input side packets to be added.
- (void)addSidePackets:(const std::map<std::string, mediapipe::Packet> &)extraSidePackets;

// TODO: rename to addDelegateOutputStream:packetType:
/// Add an output stream in the graph from which the delegate wants to receive
/// output. The delegate method called depends on the provided packetType.
/// @param outputStreamName The name of the output stream from which
///                         the delegate will receive frames.
/// @param packetType The type of packet provided by the output streams.
- (void)addFrameOutputStream:(const std::string &)outputStreamName
            outputPacketType:(MPPPacketType)packetType;

/// Starts running the graph.
/// @return YES if successful.
- (BOOL)startWithError:(NSError **)error;

/// Sends a generic packet into a graph input stream.
/// The graph must have been started before calling this.
/// Returns YES if the packet was successfully sent.
- (BOOL)sendPacket:(const mediapipe::Packet &)packet
        intoStream:(const std::string &)streamName
             error:(NSError **)error;

- (BOOL)movePacket:(mediapipe::Packet &&)packet
        intoStream:(const std::string &)streamName
             error:(NSError **)error;

/// Sets the maximum queue size for a stream. Experimental feature, currently
/// only supported for graph input streams. Should be called before starting the
/// graph.
- (BOOL)setMaxQueueSize:(int)maxQueueSize
              forStream:(const std::string &)streamName
                  error:(NSError **)error;

/// Creates a MediaPipe packet wrapping the given pixelBuffer;
- (mediapipe::Packet)packetWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                packetType:(MPPPacketType)packetType;

/// Sends a pixel buffer into a graph input stream, using the specified packet
/// type. The graph must have been started before calling this. Drops frames and
/// returns NO if maxFramesInFlight is exceeded. If allowOverwrite is set to YES,
/// allows MediaPipe to overwrite the packet contents on successful sending for
/// possibly increased efficiency. Returns YES if the packet was successfully sent.
- (BOOL)sendPixelBuffer:(CVPixelBufferRef)imageBuffer
             intoStream:(const std::string &)inputName
             packetType:(MPPPacketType)packetType
              timestamp:(const mediapipe::Timestamp &)timestamp
         allowOverwrite:(BOOL)allowOverwrite;

/// Sends a pixel buffer into a graph input stream, using the specified packet
/// type. The graph must have been started before calling this. Drops frames and
/// returns NO if maxFramesInFlight is exceeded. Returns YES if the packet was
/// successfully sent.
- (BOOL)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer
             intoStream:(const std::string &)inputName
             packetType:(MPPPacketType)packetType
              timestamp:(const mediapipe::Timestamp &)timestamp;

/// Sends a pixel buffer into a graph input stream, using the specified packet
/// type. The graph must have been started before calling this. Drops frames and
/// returns NO if maxFramesInFlight is exceeded. If allowOverwrite is set to YES,
/// allows MediaPipe to overwrite the packet contents on successful sending for
/// possibly increased efficiency. Returns YES if the packet was successfully sent.
/// Sets error to a non-nil value if an error occurs in the graph when sending the
/// packet.
- (BOOL)sendPixelBuffer:(CVPixelBufferRef)imageBuffer
             intoStream:(const std::string &)inputName
             packetType:(MPPPacketType)packetType
              timestamp:(const mediapipe::Timestamp &)timestamp
         allowOverwrite:(BOOL)allowOverwrite
                  error:(NSError **)error;

/// Sends a pixel buffer into a graph input stream, using the specified packet
/// type. The graph must have been started before calling this. The timestamp is
/// automatically incremented from the last timestamp used by this method. Drops
/// frames and returns NO if maxFramesInFlight is exceeded. Returns YES if the
/// packet was successfully sent.
- (BOOL)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer
             intoStream:(const std::string &)inputName
             packetType:(MPPPacketType)packetType;

/// Cancels a graph run. You must still call waitUntilDoneWithError: after this.
- (void)cancel;

/// Check if the graph contains this input stream
- (BOOL)hasInputStream:(const std::string &)inputName;

/// Closes an input stream.
/// You must close all graph input streams before stopping the graph.
/// @return YES if successful.
- (BOOL)closeInputStream:(const std::string &)inputName error:(NSError **)error;

/// Closes all graph input streams.
/// @return YES if successful.
- (BOOL)closeAllInputStreamsWithError:(NSError **)error;

/// Stops running the graph.
/// Call this before releasing this object. All input streams must have been
/// closed. This call does not time out, so you should not call it from the main
/// thread.
/// @return YES if successful.
- (BOOL)waitUntilDoneWithError:(NSError **)error;

/// Waits for the graph to become idle.
- (BOOL)waitUntilIdleWithError:(NSError **)error;

@end
