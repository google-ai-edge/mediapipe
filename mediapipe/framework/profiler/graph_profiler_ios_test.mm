
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

#import <UIKit/UIKit.h>
#import <XCTest/XCTest.h>

#include "absl/memory/memory.h"
#include "mediapipe/framework/profiler/graph_profiler.h"
#include "mediapipe/framework/profiler/profiler_resource_util.h"
#include "mediapipe/objc/MPPGraph.h"
#include "mediapipe/objc/MPPGraphTestBase.h"

static NSString* const kTraceFilename = @"mediapipe_trace_0.binarypb";

static const char* kOutputStream = "counter";

@interface GraphProfilerTest : MPPGraphTestBase
@end

@implementation GraphProfilerTest

- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {
  XCTAssertTrue(streamName == kOutputStream);
  NSLog(@"Received counter packet.");
}

- (void)testDefaultTraceLogPathValueIsSet {
  mediapipe::CalculatorGraphConfig graphConfig;
  mediapipe::CalculatorGraphConfig::Node* node = graphConfig.add_node();
  node->set_calculator("SimpleCalculator");
  node->add_output_stream(kOutputStream);

  mediapipe::ProfilerConfig* profilerConfig = graphConfig.mutable_profiler_config();
  profilerConfig->set_trace_enabled(true);
  profilerConfig->set_enable_profiler(true);
  profilerConfig->set_trace_log_disabled(false);

  MPPGraph* graph = [[MPPGraph alloc] initWithGraphConfig:graphConfig];
  [graph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypeRaw];
  graph.delegate = self;

  NSError* error;
  BOOL success = [graph startWithError:&error];
  XCTAssertTrue(success, @"%@", error.localizedDescription);

  // Shut down the graph.
  success = [graph waitUntilDoneWithError:&error];
  XCTAssertTrue(success, @"%@", error.localizedDescription);

  mediapipe::StatusOr<string> getTraceLogDir = mediapipe::GetDefaultTraceLogDirectory();
  XCTAssertTrue(getTraceLogDir.ok(), "GetDefaultTraceLogDirectory failed.");

  NSString* directoryPath = [NSString stringWithCString:(*getTraceLogDir).c_str()
                                               encoding:[NSString defaultCStringEncoding]];
  NSString* traceLogPath = [directoryPath stringByAppendingPathComponent:kTraceFilename];
  BOOL traceLogFileExists = [[NSFileManager defaultManager] fileExistsAtPath:traceLogPath];
  XCTAssertTrue(traceLogFileExists, @"Trace log file not found at path: %@", traceLogPath);
}

@end
