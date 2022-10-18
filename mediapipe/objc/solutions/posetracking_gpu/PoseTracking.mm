#include "PoseTracking.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPTimestampConverter.h"


static const char* kVideoQueueLabel = "com.google.mediapipe.example.videoQueue";
static const char* kLandmarksOutputStream = "pose_landmarks";



# pragma  mark - PoseTrackingGraphDelegate Interface
@interface PoseTrackingGraphDelegate : NSObject<MPPGraphDelegate>
// Receives CVPixelBufferRef from the MediaPipe graph. Invoked on a MediaPipe worker thread.
@property (nonatomic) MPPGraph* mediapipeGraph;
@property (nonatomic) const char* graphOutputStream;
@property (nonatomic) MPPLayerRenderer* renderer;
@property(nonatomic) void(^poseTrackingResultsListener)(PoseTrackingResults*);

-(id) initWithMediapipeGraph: (MPPGraph*) graph graphOutputStream: (const char*) graphOutputStream
                    renderer: (MPPLayerRenderer*) renderer;
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
            fromStream:(const std::string&)streamName ;
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
            fromStream:(const std::string&)streamName ;

@end

# pragma  mark - PoseTrackingGraphDelegate Implementation

@implementation PoseTrackingGraphDelegate

-(id) initWithMediapipeGraph: (MPPGraph*) graph graphOutputStream: (const char*) graphOutputStream
    renderer: (MPPLayerRenderer*) renderer
{
    
    self.mediapipeGraph = graph;
    self.graphOutputStream =graphOutputStream;
    self.renderer = renderer;
    return self;
}

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
        self.poseTrackingResultsListener(nil);
      return;
    }
    const auto& landmarks = packet.Get<::mediapipe::NormalizedLandmarkList>();
      NSMutableArray<PoseLandmark*>* poseLandmarks =   [[NSMutableArray<PoseLandmark*> alloc] init];
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
        
        [poseLandmarks addObject: [[PoseLandmark alloc] initWithX:landmarks.landmark(i).x() y:landmarks.landmark(i).y() z:landmarks.landmark(i).z() presence:landmarks.landmark(i).presence() visibility:landmarks.landmark(i).visibility()] ];
    }
      PoseTrackingResults* results = [[PoseTrackingResults alloc] initWithLandmarks:poseLandmarks];
      self.poseTrackingResultsListener(results);
  }
    
}

@end


@interface PoseTracking(){
    // The MediaPipe graph currently in use. Initialized in viewDidLoad, started in
    // viewWillAppear: and sent video frames on videoQueue.
    MPPGraph* mediapipeGraph;
    PoseTrackingGraphDelegate* poseTrackingGraphDelegate;
    //// Helps to convert timestamp.
    MPPTimestampConverter* timestampConverter;
}

@end

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

    self->timestampConverter = [[MPPTimestampConverter alloc] init];
    
    dispatch_queue_attr_t qosAttribute = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, /*relative_priority=*/0);
    self.videoQueue = dispatch_queue_create(kVideoQueueLabel, qosAttribute);
    
    self.poseTrackingOptions = poseTrackingOptions;
    self.graphName = @"pose_tracking_gpu";
    self->mediapipeGraph = [[self class] loadGraphFromResource: self.graphName];
    self.graphInputStream = "input_video";
    
    
    if (poseTrackingOptions.showLandmarks){
        self.graphOutputStream = "output_video";
    }else{
        self.graphOutputStream = "throttled_input_video";
    }
    
    [self->mediapipeGraph addFrameOutputStream:self.graphOutputStream
                             outputPacketType:MPPPacketTypePixelBuffer];
    
    self.poseTrackingResultsListener = ^(PoseTrackingResults*){};

    
    [self->mediapipeGraph addFrameOutputStream:"pose_landmarks"
                           outputPacketType:MPPPacketTypeRaw];
    self-> poseTrackingGraphDelegate = [[PoseTrackingGraphDelegate alloc] initWithMediapipeGraph:self->mediapipeGraph graphOutputStream:self.graphOutputStream renderer:self.renderer];
    // To prevent ARC from causing an accidental memory leak in the next block
    __weak PoseTracking* weakSelf = self;
    self -> poseTrackingGraphDelegate.poseTrackingResultsListener =  ^(PoseTrackingResults* results){
        
        weakSelf.poseTrackingResultsListener(results);
    };
    self->mediapipeGraph.delegate = self->poseTrackingGraphDelegate;
    
    
    
    return self;
}



- (void)startGraph {
  // Start running self.mediapipeGraph.
  NSError* error;
  if (![self->mediapipeGraph startWithError:&error]) {
    NSLog(@"Failed to start graph: %@", error);
  }
  else if (![self->mediapipeGraph waitUntilIdleWithError:&error]) {
    NSLog(@"Failed to complete graph initial run: %@", error);
  }
}

- (void) startWithCamera: (MPPCameraInputSource*) cameraSource {
    [cameraSource setDelegate:self queue:self.videoQueue];

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
    
    self.timeStamp = timestamp;

  [self->mediapipeGraph sendPixelBuffer:imageBuffer
                            intoStream:self.graphInputStream
                            packetType:MPPPacketTypePixelBuffer
                             timestamp:[self->timestampConverter timestampForMediaTime:timestamp]];
}

#pragma mark - MPPGraphDelegate methods


@end
