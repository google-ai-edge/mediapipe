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

#import "ObjcppLib.h"
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPLayerRenderer.h"

#include "mediapipe/examples/facial_search/embeddings.h"
#include "mediapipe/examples/facial_search/labels.h"
#include "mediapipe/framework/formats/classification.pb.h"

static NSString* const kGraphName = @"facial_search_gpu";
static const char* kInputStream = "input_frame";
static const char* kMemesOutputStream = "memes";

@interface FacialSearch () <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@end

@implementation FacialSearch {
}

#pragma mark - Cleanup methods

- (void)dealloc {
  self.mediapipeGraph.delegate = nil;
  [self.mediapipeGraph cancel];
  // Ignore errors since we're cleaning up.
  [self.mediapipeGraph closeAllInputStreamsWithError:nil];
  [self.mediapipeGraph waitUntilDoneWithError:nil];
}

#pragma mark - MediaPipe graph methods

+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
  // Load the graph config resource.
  NSError* configLoadError = nil;
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  if (!resource || resource.length == 0) {
    return nil;
  }
  NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
  NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
  if (!data) {
    NSLog(@"Failed to load MediaPipe graph config: %@", configLoadError);
    return nil;
  }

  // Parse the graph config resource into mediapipe::CalculatorGraphConfig proto object.
  mediapipe::CalculatorGraphConfig config;
  config.ParseFromArray(data.bytes, data.length);

  // Create MediaPipe graph with mediapipe::CalculatorGraphConfig proto object.
  MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];

  // Add meme collection as side packets.
  const auto labels = ::mediapipe::MyCollectionLabels();
  const auto collection = ::mediapipe::MyEmbeddingsCollection();
  [newGraph
      addSidePackets:{
                         {"collection_labels", ::mediapipe::MakePacket<decltype(labels)>(labels)},
                         {"embeddings_collection",
                          ::mediapipe::MakePacket<decltype(collection)>(collection)},
                     }];

  [newGraph addFrameOutputStream:kMemesOutputStream outputPacketType:MPPPacketTypeRaw];
  return newGraph;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
    self.mediapipeGraph.delegate = self;
    // // Set maxFramesInFlight to a small value to avoid memory contention
    // // for real-time processing.
    // self.mediapipeGraph.maxFramesInFlight = 2;
    NSLog(@"inited graph %@", kGraphName);
  }
  return self;
}

- (void)startGraph {
  NSError* error;
  if (![self.mediapipeGraph startWithError:&error]) {
    NSLog(@"Failed to start graph: %@", error);
  }
  NSLog(@"Started graph %@", kGraphName);
}

#pragma mark - MPPGraphDelegate methods

// Receives CVPixelBufferRef from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string&)streamName {
  NSLog(@"recv pixelBuffer from %@", @(streamName.c_str()));
}

// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
       didOutputPacket:(const ::mediapipe::Packet&)packet
            fromStream:(const std::string&)streamName {
  NSLog(@"recv packet @%@ from %@", @(packet.Timestamp().DebugString().c_str()),
        @(streamName.c_str()));
  if (streamName == kMemesOutputStream) {
    if (packet.IsEmpty()) {
      return;
    }
    const auto& memes = packet.Get<std::vector<::mediapipe::Classification>>();
    if (memes.empty()) {
      return;
    }
    NSMutableArray<Classification*>* result = [NSMutableArray array];
    for (const auto& meme : memes) {
      auto* c = [[Classification alloc] init];
      if (c) {
        c.index = meme.index();
        c.score = meme.score();
        c.label = @(meme.label().c_str());
        NSLog(@"\tmeme %f: %@", c.score, c.label);
        [result addObject:c];
      }
    }
    NSLog(@"calling didReceive with %lu memes", memes.size());
    [_delegate didReceive:result];
  }
}

#pragma mark - MPPInputSourceDelegate methods

- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer {
  const auto ts =
      mediapipe::Timestamp(self.timestamp++ * mediapipe::Timestamp::kTimestampUnitsPerSecond);
  NSError* err = nil;
  NSLog(@"sending imageBuffer @%@ to %s", @(ts.DebugString().c_str()), kInputStream);
  auto sent = [self.mediapipeGraph sendPixelBuffer:imageBuffer
                                        intoStream:kInputStream
                                        packetType:MPPPacketTypePixelBuffer
                                         timestamp:ts
                                    allowOverwrite:NO
                                             error:&err];
  NSLog(@"imageBuffer %s", sent ? "sent!" : "not sent.");
  if (err) {
    NSLog(@"sendPixelBuffer error: %@", err);
  }
}

@end
