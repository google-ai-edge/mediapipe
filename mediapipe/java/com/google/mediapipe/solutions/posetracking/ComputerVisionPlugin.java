package com.google.mediapipe.solutions.posetracking;

public interface ComputerVisionPlugin {
    void bodyJoints(int timestamp, BodyJoints bodyJoints);
}
