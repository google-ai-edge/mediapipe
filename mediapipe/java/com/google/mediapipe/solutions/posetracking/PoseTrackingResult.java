// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.solutions.posetracking;

import static java.lang.Math.min;

import android.graphics.Bitmap;

import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.formats.proto.DetectionProto.Detection;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.solutioncore.ImageSolutionResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * FaceDetectionResult contains the detected faces, and the input {@link Bitmap} or {@link
 * TextureFrame}. If not in static image mode, the timestamp field will be set to the timestamp of
 * the corresponding input image.
 */
public class PoseTrackingResult extends ImageSolutionResult {
    private final ImmutableList<Detection> multiPoseDetections;
    private final ImmutableList<LandmarkProto.Landmark> multiPoseLandmarks;


    public static final int NOSE = 0;
    public static final int LEFT_EYE_INNER = 1;
    public static final int LEFT_EYE = 2;
    public static final int LEFT_EYE_OUTER = 3;
    public static final int RIGHT_EYE_INNER = 4;
    public static final int RIGHT_EYE = 5;
    public static final int RIGHT_EYE_OUTER = 6;
    public static final int LEFT_EAR = 7;
    public static final int RIGHT_EAR = 8;
    public static final int MOUTH_LEFT = 9;
    public static final int MOUTH_RIGHT = 10;
    public static final int LEFT_SHOULDER = 11;
    public static final int RIGHT_SHOULDER = 12;
    public static final int LEFT_ELBOW = 13;
    public static final int RIGHT_ELBOW = 14;
    public static final int LEFT_WRIST = 15;
    public static final int RIGHT_WRIST = 16;
    public static final int LEFT_PINKY = 17;
    public static final int RIGHT_PINKY = 18;
    public static final int LEFT_INDEX = 19;
    public static final int RIGHT_INDEX = 20;
    public static final int LEFT_THUMB = 21;
    public static final int RIGHT_THUMB = 22;
    public static final int LEFT_HIP = 23;
    public static final int RIGHT_HIP = 24;
    public static final int LEFT_KNEE = 25;
    public static final int RIGHT_KNEE = 26;
    public static final int LEFT_ANKLE = 27;
    public static final int RIGHT_ANKLE = 28;
    public static final int LEFT_HEEL = 29;
    public static final int RIGHT_HEEL = 30;
    public static final int LEFT_FOOT = 31;
    public static final int RIGHT_FOOT = 32;

    // Additional points not provided by MediaPipe
    public static final int PELVIS = 33;
    public static final int SPINE = 34;
    public static final int THORAX = 35;
    public static final int HEAD_TOP = 36;


    PoseTrackingResult(
            ImmutableList<Detection> multiPoseDetections, ImmutableList<LandmarkProto.Landmark> multiPoseLandmarks, Packet imagePacket, long timestamp) {
        this.multiPoseDetections = multiPoseDetections;
        this.multiPoseLandmarks = multiPoseLandmarks;
        this.timestamp = timestamp;
        this.imagePacket = imagePacket;
    }

    // Collection of detected faces, where each face is represented as a detection proto message that
    // contains a bounding box and 6 {@link FaceKeypoint}s. The bounding box is composed of xmin and
    // width (both normalized to [0.0, 1.0] by the image width) and ymin and height (both normalized
    // to [0.0, 1.0] by the image height). Each keypoint is composed of x and y, which are normalized
    // to [0.0, 1.0] by the image width and height respectively.
    public ImmutableList<Detection> multiPoseTrackings() {
        return multiPoseDetections;
    }

    static LandmarkProto.Landmark getJointBetweenPoints(LandmarkProto.Landmark pt1, LandmarkProto.Landmark pt2, float distance) {
        return LandmarkProto.Landmark.newBuilder()
                .setX(pt1.getX() + (pt2.getX() - pt1.getX()) * distance)
                .setY(pt1.getY() + (pt2.getY() - pt1.getY()) * distance)
                .setZ(pt1.getZ() + (pt2.getZ() - pt1.getZ()) * distance)
                .setPresence(min(pt1.getPresence(), pt2.getPresence()))
                .setVisibility(min(pt1.getVisibility(), pt2.getVisibility())).build();
    }

    LandmarkProto.Landmark getPelvis(ImmutableList<LandmarkProto.Landmark> landmarks) {
        // middle point b/w left hip and right hip
        return getJointBetweenPoints(landmarks.get(LEFT_HIP), landmarks.get(RIGHT_HIP), 0.5f);
    }

    LandmarkProto.Landmark getSpinePoint(ImmutableList<LandmarkProto.Landmark> landmarks, float distanceFromShoulders) {
        LandmarkProto.Landmark pelvis = getPelvis(landmarks);
        // middle point b/w left shoulder and right shoulder
        LandmarkProto.Landmark chest = getJointBetweenPoints(landmarks.get(LEFT_SHOULDER), landmarks.get(RIGHT_SHOULDER), 0.5f);

        return getJointBetweenPoints(chest, pelvis, distanceFromShoulders);
    }

    LandmarkProto.Landmark getHeadTop(ImmutableList<LandmarkProto.Landmark> landmarks) {

        float x = 0;
        float y  = 0;
        float z = 0;
        final List<Integer> ptsIdx = Arrays.asList(LEFT_EAR, LEFT_EYE, RIGHT_EYE, RIGHT_EAR);
        for (Integer i :ptsIdx){
            LandmarkProto.Landmark landmark = landmarks.get(i);
            x+=landmark.getX();
            y+=landmark.getY();
            z+=landmark.getZ();
        }
        x = x/ptsIdx.size();
        y = y/ptsIdx.size();
        z = z/ptsIdx.size();
        LandmarkProto.Landmark midupper = LandmarkProto.Landmark.newBuilder().setX(x).setY(y).setZ(z).build();


        LandmarkProto.Landmark midlower = getJointBetweenPoints(landmarks.get(MOUTH_LEFT), landmarks.get(MOUTH_RIGHT), 0.5f);
        // 2 times the distance b/w nose and eyes
        return getJointBetweenPoints(midlower, midupper, 2.5f);
    }

    ImmutableList<LandmarkProto.Landmark> getAdditionalLandmarksByInterpolation(ImmutableList<LandmarkProto.Landmark> originalLandmarks) {

        if (originalLandmarks.isEmpty()) return originalLandmarks;
        List<LandmarkProto.Landmark> landmarks = new ArrayList<>(originalLandmarks);
        // pelvis
        landmarks.add(getPelvis(originalLandmarks));
        // spine assuming it is 2/3rd of distance b/w shoulders and pelvis
        landmarks.add(getSpinePoint(originalLandmarks, 2 / 3f));
        // thorax assuming it is 1/3rd of distance b/w shoulders and pelvis
        landmarks.add(getSpinePoint(originalLandmarks, 1 / 3f));
        // head top
        landmarks.add(getHeadTop(originalLandmarks));


        return ImmutableList.copyOf(landmarks);

    }

    public ImmutableList<LandmarkProto.Landmark> multiPoseLandmarks() {
        return getAdditionalLandmarksByInterpolation(multiPoseLandmarks);
    }

    public static Builder builder() {
        return new AutoBuilder_PoseTrackingResult_Builder();
    }

    /**
     * Builder for {@link PoseTrackingResult}.
     */
    @AutoBuilder
    public abstract static class Builder {
        abstract Builder setMultiPoseDetections(List<Detection> value);

        abstract Builder setMultiPoseLandmarks(List<LandmarkProto.Landmark> value);

        abstract Builder setTimestamp(long value);

        abstract Builder setImagePacket(Packet value);

        abstract PoseTrackingResult build();
    }
}
