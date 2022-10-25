//
// Created by Mautisim Munir on 05/10/2022.
//

#ifndef MEDIAPIPE_POSETRACKINGOPTIONS_H
#define MEDIAPIPE_POSETRACKINGOPTIONS_H
#import <Foundation/Foundation.h>



@interface PoseTrackingOptions: NSObject
@property(nonatomic) int modelComplexity;
@property(nonatomic) bool showLandmarks;
//@property(nonatomic) int cameraRotation;


- (instancetype) initWithShowLandmarks : (bool) showLandmarks modelComplexity:  (int) modelComplexity;

@end

#endif //MEDIAPIPE_POSETRACKINGOPTIONS_H
