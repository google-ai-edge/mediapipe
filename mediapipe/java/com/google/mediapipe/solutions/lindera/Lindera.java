package com.google.mediapipe.solutions.lindera;

import android.util.Log;
import android.view.View;
import android.view.ViewGroup;

import androidx.appcompat.app.AppCompatActivity;

import com.google.common.collect.ImmutableList;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutions.posetracking.PoseTracking;
import com.google.mediapipe.solutions.posetracking.PoseTrackingOptions;
import com.google.mediapipe.solutions.posetracking.PoseTrackingResult;
import com.google.mediapipe.solutions.posetracking.PoseTrackingResultGlRenderer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Lindera {
    
    private ComputerVisionPlugin plugin;
    public FpsHelper fpsHelper = new FpsHelper();
    private PoseTracking poseTracking;
    

    private long getTimeStamp(){
        return System.nanoTime();
    }
    private CameraRotation cameraRotation = CameraRotation.AUTOMATIC;
    
    // Live camera demo UI and camera components.
    private CameraInput cameraInput;
    private SolutionGlSurfaceView<PoseTrackingResult> glSurfaceView;
    private CameraInput.CameraFacing cameraFacing = CameraInput.CameraFacing.FRONT;
    private AppCompatActivity appCompatActivity;
    private  ViewGroup computerVisionContainerView;
    private PoseTrackingResultGlRenderer solutionRenderer;
    
    public Lindera(ComputerVisionPlugin plugin){
        this.plugin = plugin;
    }

    public void setLandmarksVisibility(boolean visible){
        this.poseTracking.options = PoseTrackingOptions.builder().withPoseTrackingOptions(this.poseTracking
                .options).setLandmarkVisibility(visible).build();
        solutionRenderer.setLandmarksVisibility(this.poseTracking.options.landmarkVisibility());
        glSurfaceView.setSolutionResultRenderer(solutionRenderer);
    }
    public boolean getLandmarkVisibility(){
        return this.poseTracking.options.landmarkVisibility();
    }

    public int getModelComplexity(){
        return this.poseTracking.options.modelComplexity();

    }
    public void setModelComplexity(int complexity){
        this.poseTracking.options = PoseTrackingOptions.builder().withPoseTrackingOptions(this.poseTracking
                .options).setModelComplexity(complexity).build();
    }

    public void restartDetection(){
        if (poseTracking!=null) {
            stopDetection();
            startDetection(poseTracking.options);
        }else{
            startDetection();
        }

    }

    public void initialize (ViewGroup computerVisionContainerView , AppCompatActivity appCompatActivity){

        this.computerVisionContainerView = computerVisionContainerView;
        this.appCompatActivity = appCompatActivity;
        startDetection();
    }

    public void setCameraRotation(CameraRotation cameraRotation){
        this.cameraRotation = cameraRotation;
    }

    private void setupEventListener() {
        poseTracking.setResultListener(
            poseTrackingResult -> {
                fpsHelper.logNewPoint();
                glSurfaceView.setRenderData(poseTrackingResult);
                glSurfaceView.requestRender();
                ImmutableList<LandmarkProto.Landmark> landmarks = poseTrackingResult.multiPoseLandmarks();

                if (landmarks.isEmpty()) return;

                BodyJoints bodyJoints = new BodyJoints();
                landmarksToBodyJoints(landmarks,bodyJoints);

                plugin.bodyJoints(getTimeStamp(), bodyJoints);
            });
    }

    public List<String> getAvailableCameras(){
        return Arrays.stream(CameraInput.CameraFacing.values()).map(Enum::name).collect(Collectors.toList());

    }

    public void doOnDestroy(){
        stopDetection();
        appCompatActivity = null;
        computerVisionContainerView = null;

    }

    /**
     * Will need to restart camera pipeline if already started
     * @param name One of FRONT or BACK
     */
    public void setCamera(String name){
        cameraFacing = CameraInput.CameraFacing.valueOf(name);

    }

    /**
     * To satisfy original API we start detection with reasonable 
     * default setting 
     * */    
    public void startDetection() {
        startDetection(
            PoseTrackingOptions.builder()
                .setStaticImageMode(false)
                .setLandmarkVisibility(false)
                .setModelComplexity(1)
                .setSmoothLandmarks(true)
                .build()
                );    
    }

    public void startDetection(PoseTrackingOptions options){
        // ensure that class is initalized
        assert (appCompatActivity != null);
        // Initializes a new MediaPipe Face Detection solution instance in the streaming mode.
        poseTracking =
                new PoseTracking(
                        appCompatActivity,
                        options);
        poseTracking.setErrorListener(
                (message, e) -> Log.e("Lindera", "MediaPipe Pose Tracking error:" + message));
        cameraInput = new CameraInput(appCompatActivity);

        cameraInput.setNewFrameListener(textureFrame -> poseTracking.send(textureFrame));

        // Initializes a new Gl surface view with a user-defined PoseTrackingResultGlRenderer.
        glSurfaceView = 
            new SolutionGlSurfaceView<>(
                appCompatActivity, 
                poseTracking.getGlContext(), 
                poseTracking.getGlMajorVersion()
            );
        solutionRenderer = new PoseTrackingResultGlRenderer();
        solutionRenderer.setLandmarksVisibility(this.poseTracking.options.landmarkVisibility());
        glSurfaceView.setSolutionResultRenderer(solutionRenderer);
        glSurfaceView.setRenderInputImage(true);

        setupEventListener();
        
        // The runnable to start camera after the gl surface view is attached.
        // For video input source, videoInput.start() will be called when the video uri is available.
        glSurfaceView.post(this::startCamera);
        // Updates the preview layout.

        computerVisionContainerView.removeAllViewsInLayout();
        computerVisionContainerView.addView(glSurfaceView);
        glSurfaceView.setVisibility(View.VISIBLE);
        computerVisionContainerView.requestLayout();
    }

    public void stopDetection(){
        if (cameraInput != null) {
            cameraInput.setNewFrameListener(null);
            cameraInput.close();
        }
        if (glSurfaceView != null) {
            glSurfaceView.setVisibility(View.GONE);
        }
        if (poseTracking != null) {
            poseTracking.close();
        }
    }

    private void landmarkToXYZPointWithConfidence(LandmarkProto.Landmark landmark,XYZPointWithConfidence bodyJoint){
        bodyJoint.x = landmark.getX();
        bodyJoint.y = landmark.getY();
        bodyJoint.z = landmark.getZ();
        bodyJoint.confidence = landmark.getVisibility();
        bodyJoint.presence = landmark.getPresence();

    }

    private void landmarksToBodyJoints(ImmutableList<LandmarkProto.Landmark> landmarks , BodyJoints bodyJoints){
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.NOSE), bodyJoints.neckNose);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_EYE_INNER), bodyJoints.leftEyeInner);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_EYE), bodyJoints.leftEye);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_EYE_OUTER), bodyJoints.leftEyeOuter);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_EYE_INNER), bodyJoints.rightEyeInner);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_EYE), bodyJoints.rightEye);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_EYE_OUTER), bodyJoints.rightEyeOuter);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_EAR), bodyJoints.leftEar);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_EAR), bodyJoints.rightEar);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.MOUTH_LEFT), bodyJoints.mouthLeft);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.MOUTH_RIGHT), bodyJoints.mouthRight);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_SHOULDER), bodyJoints.leftShoulder);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_SHOULDER), bodyJoints.rightShoulder);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_ELBOW), bodyJoints.leftElbow);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_ELBOW), bodyJoints.rightElbow);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_WRIST), bodyJoints.leftWrist);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_WRIST), bodyJoints.rightWrist);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_PINKY), bodyJoints.leftPinky);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_PINKY), bodyJoints.rightPinky);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_INDEX), bodyJoints.leftIndex);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_INDEX), bodyJoints.rightIndex);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_THUMB), bodyJoints.leftThumb);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_THUMB), bodyJoints.rightThumb);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_HIP), bodyJoints.leftHip);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_HIP), bodyJoints.rightHip);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_KNEE), bodyJoints.leftKnee);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_KNEE), bodyJoints.rightKnee);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_ANKLE), bodyJoints.rightAnkle);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_ANKLE), bodyJoints.leftAnkle);


        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_HEEL), bodyJoints.rightHeel);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_HEEL), bodyJoints.leftHeel);

        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.RIGHT_FOOT), bodyJoints.rightFoot);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.LEFT_FOOT), bodyJoints.leftFoot);

        // additional points
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.PELVIS), bodyJoints.pelvis);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.SPINE), bodyJoints.spine);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.THORAX), bodyJoints.thorax);
        landmarkToXYZPointWithConfidence(landmarks.get(PoseTrackingResult.HEAD_TOP), bodyJoints.headTop);




    }

    private void startCamera() {
        if (cameraRotation!=CameraRotation.AUTOMATIC) {
            cameraInput.getConverter(poseTracking.getGlContext()).setRotation(cameraRotation.getValue());
        }
        cameraInput.start(
                appCompatActivity,
                poseTracking.getGlContext(),
                cameraFacing,
                glSurfaceView.getWidth(),
                glSurfaceView.getHeight());
    }

}
