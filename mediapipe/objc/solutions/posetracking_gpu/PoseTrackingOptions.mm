#import "PoseTrackingOptions.h"

@implementation PoseTrackingOptions

- (instancetype) initWithShowLandmarks : (bool) showLandmarks {
//    self.cameraRotation = cameraRotation;
    self.showLandmarks = showLandmarks;
    return self;
}


@end
