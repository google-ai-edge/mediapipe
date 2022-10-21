#import "PoseTrackingOptions.h"

@implementation PoseTrackingOptions

- (instancetype) initWithShowLandmarks : (bool) showLandmarks modelComplexity:  (int) modelComplexity{
//    self.cameraRotation = cameraRotation;
    self.showLandmarks = showLandmarks;
    self.modelComplexity = modelComplexity;
    return self;
}


@end
