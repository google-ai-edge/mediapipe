#include "PoseTrackingResults.h"


@implementation PoseLandmark

- (instancetype) initWithX: (float) x y:(float) y z:(float) z presence:(float) presence visibility:(float) visibility{
    self.x = x;
    self.y = y;
    self.z = z;
    self.presence = presence;
    self.visibility = visibility;
    return self;
}

@end


@implementation PoseTrackingResults

- (instancetype) initWithLandmarks: (NSArray<PoseLandmark*>*) landmarks{
    self.landmarks = landmarks;
    return self;
}

@end
