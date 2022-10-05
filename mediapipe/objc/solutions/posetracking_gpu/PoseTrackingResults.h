#ifndef MEDIAPIPE_POSETRACKINGRESULTS_H
#define MEDIAPIPE_POSETRACKINGRESULTS_H

#import <Foundation/Foundation.h>
@interface PoseLandmark: NSObject

@property float x;
@property float y;
@property float z;
@property float presence;
@property float visibility;

- (instancetype) initWithX: (float) x y:(float) y z:(float) z presence:(float) presence visibility:(float) visibility;

@end

@interface PoseTrackingResults : NSObject


@property NSArray<PoseLandmark*>* landmarks;

- (instancetype) initWithLandmarks: (NSArray<PoseLandmark*>*) landmarks;

@end

#endif //MEDIAPIPE_POSETRACKINGRESULTS_H
