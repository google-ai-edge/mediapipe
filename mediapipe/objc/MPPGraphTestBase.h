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

#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/NSError+util_status.h"
#import "mediapipe/objc/util.h"

/// This XCTestCase subclass provides common convenience methods for testing
/// with MPPGraph.
@interface MPPGraphTestBase : XCTestCase <MPPGraphDelegate> {
  /// This block is used to respond to mediapipeGraph:didOutputPixelBuffer:fromStream:.
  /// runGraph:withPixelBuffer:packetType: uses this internally, but you can reuse it
  /// if you need to run a graph directly and want a MPPGraphTestBase object to
  /// act as the delegate.
  void (^_pixelBufferOutputBlock)(MPPGraph* graph, CVPixelBufferRef imageBuffer,
                                  const std::string& streamName);

  /// This block is used to respond to mediapipeGraph:didOutputPacket:fromStream:.
  /// You can use it if you need to run a graph directly and want a MPPGraphTestBase
  /// object to act as the delegate.
  void (^_packetOutputBlock)(MPPGraph* graph, const mediapipe::Packet& packet,
                             const std::string& streamName);
}

/// Runs a single frame through a simple graph. The graph is expected to have an
/// input stream named "input_frames" and an output stream named
/// "output_frames". This function runs the graph, sends inputBuffer into
/// input_frames (at timestamp=1), receives an output buffer from output_frames,
/// completes the run, and returns the output frame.
- (CVPixelBufferRef)runGraph:(MPPGraph*)graph
             withPixelBuffer:(CVPixelBufferRef)inputBuffer
                  packetType:(MPPPacketType)inputPacketType;

/// Runs a simple graph, providing a single frame to zero or more inputs. Input images are wrapped
/// in packets each with timestamp mediapipe::Timestamp(1). Those packets are added to the
/// designated streams (named by the keys of withInputPixelBuffers). When a packet arrives on the
/// output stream, the graph run is done and the output frame is returned.
- (CVPixelBufferRef)runGraph:(MPPGraph*)graph
       withInputPixelBuffers:
           (const std::unordered_map<std::string, CFHolder<CVPixelBufferRef>>&)inputBuffers
                outputStream:(const std::string&)output
                  packetType:(MPPPacketType)inputPacketType;

/// Loads a data file from the test bundle.
- (NSData*)testDataNamed:(NSString*)name extension:(NSString*)extension;

/// Loads an image from the test bundle.
- (UIImage*)testImageNamed:(NSString*)name extension:(NSString*)extension;

/// Returns a URL for a file.extension in the test bundle.
- (NSURL*)URLForTestFile:(NSString*)file extension:(NSString*)extension;

/// Loads an image from the test bundle in subpath.
- (UIImage*)testImageNamed:(NSString*)name
                 extension:(NSString*)extension
              subdirectory:(NSString*)subdirectory;

/// Compares two pixel buffers for strict equality.
/// Returns true iff the two buffers have the same size, format, and pixel data.
- (BOOL)pixelBuffer:(CVPixelBufferRef)a isEqualTo:(CVPixelBufferRef)b;

/// Compares two pixel buffers with some leniency.
/// Returns true iff the two buffers have the same size and format, and:
/// - the difference between each pixel of A and the corresponding pixel of B does
///   not exceed maxLocalDiff, and
/// - the average difference between corresponding pixels of A and B does not
///   exceed maxAvgDiff.
- (BOOL)pixelBuffer:(CVPixelBufferRef)a
               isCloseTo:(CVPixelBufferRef)b
      maxLocalDifference:(int)maxLocalDiff
    maxAverageDifference:(float)maxAvgDiff;

/// Compares two pixel buffers with some leniency.
/// Returns true iff the two buffers have the same size and format, and:
/// - the difference between each pixel of A and the corresponding pixel of B does
///   not exceed maxLocalDiff, and
/// - the average difference between corresponding pixels of A and B does not
///   exceed maxAvgDiff.
/// The maximum local difference and average difference will be written
/// to @c maxLocalDiffOut and @c maxAvgDiffOut respectively.
- (BOOL)pixelBuffer:(CVPixelBufferRef)a
                  isCloseTo:(CVPixelBufferRef)b
         maxLocalDifference:(int)maxLocalDiff
       maxAverageDifference:(float)maxAvgDiff
      maxLocalDifferenceOut:(int*)maxLocalDiffOut
    maxAverageDifferenceOut:(float*)maxAvgDiffOut;

/// Utility function for making a copy of a pixel buffer with a different pixel
/// format.
- (CVPixelBufferRef)convertPixelBuffer:(CVPixelBufferRef)input toPixelFormat:(OSType)pixelFormat;

/// Makes a scaled copy of a BGRA pixel buffer.
- (CVPixelBufferRef)scaleBGRAPixelBuffer:(CVPixelBufferRef)input toSize:(CGSize)size;

/// Utility function for transforming a pixel buffer.
/// It creates a new pixel buffer with the same dimensions as the original, in the
/// desired pixel format, and invokes a block with the input and output buffers.
/// The buffers are locked before the block and unlocked after, so the block can read
/// from the input buffer and write to the output buffer without further preparation.
- (CVPixelBufferRef)transformPixelBuffer:(CVPixelBufferRef)input
                       outputPixelFormat:(OSType)pixelFormat
                          transformation:(void (^)(CVPixelBufferRef input,
                                                   CVPixelBufferRef output))transformation;

/// Computes a difference image from two input images. Useful for debugging.
- (UIImage*)differenceOfImage:(UIImage*)inputA image:(UIImage*)inputB;

/// Tests a graph by sending in the provided input pixel buffer and comparing the
/// output with the provided expected output. Uses runGraph:withPixelBuffer:packetType:
/// internally, so the streams are supposed to be named "input_frames" and "output_frames".
/// The actual and expected outputs are compared fuzzily.
- (void)testGraph:(MPPGraph*)graph
             input:(CVPixelBufferRef)inputBuffer
    expectedOutput:(CVPixelBufferRef)expectedBuffer;

/// Tests a graph by sending the provided image files as pixelBuffer inputs to the
/// corresponding streams, and comparing the single frame output by the given output stream
/// with the contents of the given output file.
/// @param config Graph config.
/// @param fileInputs Dictionary mapping input stream names to image file paths.
/// @param packetInputs Map of input stream names to additional input packets.
/// @param sidePackets Map of input side packet stream names to packets.
/// @param outputStream Name of the output stream where the output is produced.
/// @param expectedPath Path to an image file containing the expected output.
/// @param maxAverageDifference The maximum allowable average pixel difference
/// between the
///        expected output and computed output.
/// TODO: Use NSDictionary instead of std::map for sidePackets.
- (void)testGraphConfig:(const mediapipe::CalculatorGraphConfig&)config
      inputStreamsAndFiles:(NSDictionary<NSString*, NSString*>*)fileInputs
    inputStreamsAndPackets:(const std::map<std::string, mediapipe::Packet>&)packetInputs
               sidePackets:(std::map<std::string, mediapipe::Packet>)sidePackets
                 timestamp:(mediapipe::Timestamp)timestamp
              outputStream:(NSString*)outputStream
        expectedOutputFile:(NSString*)expectedPath
      maxAverageDifference:(float)maxAverageDifference;

/// Calls the above testGraphConfig: method with a default maxAverageDifference
/// of 1.f and timestamp of 1.
- (void)testGraphConfig:(const mediapipe::CalculatorGraphConfig&)config
    inputStreamsAndFiles:(NSDictionary<NSString*, NSString*>*)inputs
            outputStream:(NSString*)outputStream
      expectedOutputFile:(NSString*)expectedPath;

@end
