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

#import <XCTest/XCTest.h>

#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPGraphTestBase.h"

@interface MPPGpuSimpleTest : MPPGraphTestBase
@end

@implementation MPPGpuSimpleTest{
  CFHolder<CVPixelBufferRef> _inputPixelBuffer;
  CFHolder<CVPixelBufferRef> _referencePixelBuffer;
  CFHolder<CVPixelBufferRef> _outputPixelBuffer;
}
- (void)setUp {
  [super setUp];
  UIImage* image = [self testImageNamed:@"sergey" extension:@"png"];
  XCTAssertTrue(CreateCVPixelBufferFromCGImage(image.CGImage, &_inputPixelBuffer).ok());
  image = [self testImageNamed:@"sobel_reference" extension:@"png"];
  XCTAssertTrue(CreateCVPixelBufferFromCGImage(image.CGImage, &_referencePixelBuffer).ok());
}

// This delegate method receives output.
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string&)streamName
               timestamp:(const mediapipe::Timestamp&)timestamp {
  NSLog(@"CALLBACK INVOKED");
  _outputPixelBuffer.reset(pixelBuffer);
}

- (void)testSimpleGpuGraph {
  // Graph setup.
  NSData* configData = [self testDataNamed:@"test_sobel.binarypb" extension:nil];
  mediapipe::CalculatorGraphConfig config;
  XCTAssertTrue(config.ParseFromArray([configData bytes], [configData length]));
  MPPGraph* mediapipeGraph = [[MPPGraph alloc] initWithGraphConfig:config];
  // We receive output by setting ourselves as the delegate.
  mediapipeGraph.delegate = self;
  [mediapipeGraph addFrameOutputStream:"output_video" outputPacketType:MPPPacketTypePixelBuffer];

  // Start running the graph.
  NSError *error;
  BOOL success = [mediapipeGraph startWithError:&error];
  XCTAssertTrue(success, @"%@", error.localizedDescription);

  // Send a frame.
  XCTAssertTrue([mediapipeGraph sendPixelBuffer:*_inputPixelBuffer
                                   intoStream:"input_video"
                                   packetType:MPPPacketTypePixelBuffer
                                    timestamp:mediapipe::Timestamp(0)]);

  // Shut down the graph.
  success = [mediapipeGraph closeAllInputStreamsWithError:&error];
  XCTAssertTrue(success, @"%@", error.localizedDescription);
  success = [mediapipeGraph waitUntilDoneWithError:&error];
  XCTAssertTrue(success, @"%@", error.localizedDescription);

  // Check output.
  XCTAssertTrue(_outputPixelBuffer != nullptr);
  [self savePixelBufferToSponge:*_outputPixelBuffer
                    withSubpath:@"sobel.png"];
  XCTAssertTrue([self pixelBuffer:*_outputPixelBuffer
                        isCloseTo:*_referencePixelBuffer
               maxLocalDifference:5
             maxAverageDifference:FLT_MAX]);
}
@end
