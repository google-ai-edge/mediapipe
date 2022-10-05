#import "PoseTrackingOptions.h"

@implementation PoseTrackingOptions

- (instancetype) initWithShowLandmarks : (bool) showLandmarks cameraRotation:(int) cameraRotation{
    self.cameraRotation = cameraRotation;
    self.showLandmarks = showLandmarks;
    return self;
}


@end
