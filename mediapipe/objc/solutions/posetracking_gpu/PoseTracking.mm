#include "PoseTracking.h"
#include "mediapipe/framework/formats/landmark.pb.h"

static const char* kVideoQueueLabel = "com.google.mediapipe.example.videoQueue";
static const char* kLandmarksOutputStream = "pose_landmarks";

@implementation PoseTracking

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
  return newGraph;
}

- (instancetype) initWithPoseTrackingOptions: (PoseTrackingOptions*) poseTrackingOptions{
    self.renderer = [[MPPLayerRenderer alloc] init];
    self.renderer.frameScaleMode = MPPFrameScaleModeFillAndCrop;

    self.timestampConverter = [[MPPTimestampConverter alloc] init];
    
    dispatch_queue_attr_t qosAttribute = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, /*relative_priority=*/0);
    self.videoQueue = dispatch_queue_create(kVideoQueueLabel, qosAttribute);
    
    self.poseTrackingOptions = poseTrackingOptions;
    self.graphName = @"pose_tracking_gpu";
    self.mediapipeGraph = [[self class] loadGraphFromResource: self.graphName];
    self.graphInputStream = "input_video";

    if (poseTrackingOptions.showLandmarks){
        self.graphOutputStream = "output_video";
    }else{
        self.graphOutputStream = "throttled_input_video";
    }
    
    [self.mediapipeGraph addFrameOutputStream:self.graphOutputStream
                             outputPacketType:MPPPacketTypePixelBuffer];
    
    
    
    [self.mediapipeGraph addFrameOutputStream:"pose_landmarks"
                           outputPacketType:MPPPacketTypeRaw];
    
    self.mediapipeGraph.delegate = self;
    
    
    
    
    return self;
}



- (void)startGraph {
  // Start running self.mediapipeGraph.
  NSError* error;
  if (![self.mediapipeGraph startWithError:&error]) {
    NSLog(@"Failed to start graph: %@", error);
  }
  else if (![self.mediapipeGraph waitUntilIdleWithError:&error]) {
    NSLog(@"Failed to complete graph initial run: %@", error);
  }
}

- (void) startWithCamera: (MPPCameraInputSource*) cameraSource {
    [self startGraph];
    // Start fetching frames from the camera.
    dispatch_async(self.videoQueue, ^{
      [cameraSource start];
    });
}


#pragma mark - MPPInputSourceDelegate methods

// Must be invoked on self.videoQueue.
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer
                timestamp:(CMTime)timestamp
               fromSource:(MPPInputSource*)source {

  [self.mediapipeGraph sendPixelBuffer:imageBuffer
                            intoStream:self.graphInputStream
                            packetType:MPPPacketTypePixelBuffer
                             timestamp:[self.timestampConverter timestampForMediaTime:timestamp]];
}

#pragma mark - MPPGraphDelegate methods

// Receives CVPixelBufferRef from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string&)streamName {
  if (streamName == self.graphOutputStream) {
    // Display the captured image on the screen.
    CVPixelBufferRetain(pixelBuffer);
    dispatch_async(dispatch_get_main_queue(), ^{
      [self.renderer renderPixelBuffer:pixelBuffer];
      CVPixelBufferRelease(pixelBuffer);
    });
  }
}


// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {
  if (streamName == kLandmarksOutputStream) {
    if (packet.IsEmpty()) {
      NSLog(@"[TS:%lld] No pose landmarks", packet.Timestamp().Value());
      return;
    }
    const auto& landmarks = packet.Get<::mediapipe::NormalizedLandmarkList>();
    NSLog(@"[TS:%lld] Number of pose landmarks: %d", packet.Timestamp().Value(),
          landmarks.landmark_size());
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
      NSLog(@"\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
            landmarks.landmark(i).y(), landmarks.landmark(i).z());
    }
  }
}
@end
