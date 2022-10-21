//
// Created by Mautisim Munir on 05/10/2022.
//

#ifndef MEDIAPIPE_POSETRACKING_H
#define MEDIAPIPE_POSETRACKING_H
#import <Foundation/Foundation.h>
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPLayerRenderer.h"
#import "mediapipe/objc/MPPPlayerInputSource.h"
#import "PoseTrackingOptions.h"
#import "PoseTrackingResults.h"
@interface PoseTracking : NSObject<MPPInputSourceDelegate>

// The MediaPipe graph currently in use. Initialized in viewDidLoad, started in
// viewWillAppear: and sent video frames on videoQueue.
//@property(nonatomic) MPPGraph* mediapipeGraph;




// Render frames in a layer.
@property(nonatomic) MPPLayerRenderer* renderer;

@property (nonatomic) CMTime timeStamp;
// Graph name.
@property(nonatomic) NSString* graphName;

// Graph input stream.
@property(nonatomic) const char* graphInputStream;

// Graph output stream.
@property(nonatomic) const char* graphOutputStream;

// Modify graph options
@property(nonatomic) PoseTrackingOptions* poseTrackingOptions;


// Process camera frames on this queue.
@property(nonatomic) dispatch_queue_t videoQueue;

// Codeblock that runs whenever pose tracking results are available
@property(nonatomic) void(^poseTrackingResultsListener)(PoseTrackingResults*);

- (instancetype) initWithPoseTrackingOptions: (PoseTrackingOptions*) poseTrackingOptions;
- (void)startGraph;
- (void) startWithCamera: (MPPCameraInputSource*) cameraSource;
- (void)showLandmarks: (BOOL) value;
- (BOOL) areLandmarksShown;
@end


#endif //MEDIAPIPE_POSETRACKING_H
