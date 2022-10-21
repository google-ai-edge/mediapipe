// This is the copperlabs posetracking api built in objective c
import MPPoseTracking
import UIKit


/// A helper class to run the Pose Tracking API
///  TFLite models are also loaded when you initialize this class
public final class Lindera{
    // initalize the PoseTracking api and load models
    var poseTracking:PoseTracking = PoseTracking(poseTrackingOptions: PoseTrackingOptions(showLandmarks: true,modelComplexity: 1))
    let fpsHelper = FPSHelper(smoothingFactor: 0.95)
    public func setFpsDelegate(fpsDelegate: @escaping (_ fps:Double)->Void){
        fpsHelper.onFpsUpdate = fpsDelegate;
    }
    
    // attach Mediapipe camera helper to our class
    let cameraSource = MPPCameraInputSource()

    // A delegate to handle results
    public weak var delegate: LinderaDelegate?
    
    
    public var cameraView: UIView {
        return self.linderaExerciseSession
    }
    public func showLandmarks(value:Bool){
        self.poseTracking.showLandmarks(value)
    }
    
    
    public func areLandmarksShown() -> Bool{
        return self.poseTracking.areLandmarksShown()
    }
    public func getModelComplexity() -> Int {
        return Int(self.poseTracking.poseTrackingOptions.modelComplexity);
    }
    public func setModelComplexityNow(complexity:Int){
        let poseTrackingOptions = poseTracking.poseTrackingOptions
        
        poseTrackingOptions?.modelComplexity = Int32(complexity)
        
        poseTracking = PoseTracking(poseTrackingOptions: poseTrackingOptions)
        startPoseTracking()
        startCamera()
        



    }
    
    

    private lazy var linderaExerciseSession: UIView = {
        
        // this will be the main camera view
        let liveView = UIView()
        
        startPoseTracking()
        


        return liveView
        
    }()
    private func  startPoseTracking(){
            // set camera preferences
        self.cameraSource.sessionPreset = AVCaptureSession.Preset.high.rawValue
        self.cameraSource.cameraPosition = AVCaptureDevice.Position.front
        self.cameraSource.orientation = AVCaptureVideoOrientation.portrait
        if (self.cameraSource.orientation == AVCaptureVideoOrientation.portrait){
        self.cameraSource.videoMirrored = true
        }
        // call LinderaDelegate on pose tracking results
        self.poseTracking.poseTrackingResultsListener = {[weak self] results in
            

            guard let self = self, let results = results else {
                return
            }
            
            self.delegate?.lindera(self, didDetect: .init(pose: Asensei3DPose.init(results), timestamp: CMTimeGetSeconds(self.poseTracking.timeStamp)))
        }
        self.poseTracking.graphOutputStreamListener = {[weak self] in
            self?.fpsHelper.logTime()
        }
        
        self.poseTracking.startGraph()
        // attach camera's output with poseTracking object and its videoQueue
        self.cameraSource.setDelegate(self.poseTracking, queue: self.poseTracking.videoQueue)
    }
    public required init(){
        
        
    }
    
    
    public func startCamera(_ completion: ((Result<Void, Error>) -> Void)? = nil) {
        // set our rendering layer frame according to cameraView boundry
        self.poseTracking.renderer.layer.frame = cameraView.layer.bounds
        // attach render CALayer on cameraView to render output to
        self.cameraView.layer.addSublayer(self.poseTracking.renderer.layer)
            
        self.cameraSource.requestCameraAccess(
            completionHandler: {(granted:Bool)->Void in
                if (granted){
                    self.poseTracking.videoQueue.async(execute:{ [weak self] in
                        
                        self?.cameraSource.start()

                     } )
                    completion?(.success(Void()))
                  }else{
                      
                      completion?(.failure(preconditionFailure("Camera Access Not Granted")))
                        
                }
        })
            
        
        
        
    }
    
    func stopCamera(){
        if (self.cameraSource.isRunning){
        self.poseTracking.videoQueue.async { [weak self] in
            self?.cameraSource.stop()
          }
            
        }
    }
    
    /// switches camera from front to back and vice versa
    func switchCamera(_ completion: ((Result<Void, Error>) -> Void)? = nil) {
        self.poseTracking.videoQueue.async { [weak self] in
            if let self = self {
                
                self.stopCamera()
                self.startCamera(completion)
                
                switch(self.cameraSource.cameraPosition){

                case .unspecified:
                    completion?(.failure(preconditionFailure("Unkown Camera Position")))
                case .back:
                    self.selectCamera(AVCaptureDevice.Position.front,completion)
                case .front:
                    self.selectCamera(AVCaptureDevice.Position.back,completion)
                @unknown default:
                    completion?(.failure(preconditionFailure("Unkown Camera Position")))

                }
                
               
            }
        
        }
     }
    
    /// Choose front or back camera. Must restart camera after use if already started
    public func selectCamera(_ position: AVCaptureDevice.Position, _ completion: ((Result<Void, Error>) -> Void)? = nil) {
        self.poseTracking.videoQueue.async { [weak self] in
            self?.cameraSource.cameraPosition = position
            completion?(.success(Void()))
        }
        
    }
    
    
    
    
}
public protocol LinderaDelegate: AnyObject {

    func lindera(_ lindera: Lindera, didDetect event: Asensei3DPose.Event)
}


/// Convert PoseLandmarks from PoseTrackingAPI to BodyJointDetails
func landmarkToBodyJointDetails(landmark: PoseLandmark) -> Asensei3DPose.BodyJointDetails{
    return Asensei3DPose.BodyJointDetails(position: .init(x: landmark.x, y: landmark.y, z: landmark.z), confidence: landmark.visibility)
}
// MARK: - Helpers
extension Asensei3DPose {

    init(_ pose: PoseTrackingResults) {
        
        self.nose = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_NOSE])
           
           self.leftEyeInner = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_EYE_INNER])
           self.leftEye = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_EYE])
           self.leftEyeOuter = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_EYE_OUTER])

           self.rightEyeInner = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_EYE_OUTER])
           self.rightEye = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_EYE])
           self.rightEyeOuter = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_EYE_OUTER])
           
           self.leftEar = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_EAR])
           self.rightEar = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_EAR])

           self.mouthLeft = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_MOUTH_LEFT])
           self.mouthRight = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_MOUTH_RIGHT])

           self.leftShoulder = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_SHOULDER])
           self.rightShoulder = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_SHOULDER])

           self.leftElbow = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_ELBOW])
           self.rightElbow = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_ELBOW])

           self.leftWrist = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_WRIST])
           self.rightWrist = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_WRIST])

           self.leftPinky = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_PINKY])
           self.rightPinky = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_PINKY])

           self.leftIndex = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_INDEX])
           self.rightIndex = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_INDEX])

           self.leftThumb = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_THUMB])
           self.rightThumb = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_THUMB])

           self.leftHip = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_HIP])
           self.rightHip = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_HIP])

           self.leftKnee = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_KNEE])
           self.rightKnee = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_KNEE])
           
           self.rightAnkle = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_ANKLE])
           self.leftAnkle = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_ANKLE])


           self.rightHeel = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_HEEL])
           self.leftHeel = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_HEEL])

           self.rightFoot = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_RIGHT_FOOT])
           self.leftFoot = landmarkToBodyJointDetails(landmark: pose.landmarks[POSE_LEFT_FOOT])
        
        

    }
}

//extension Asensei3DPose.Vector3D {
//
//    init(_ vector: Lindera3DVector) {
//        self.x = -vector.x
//        self.y = vector.z
//        self.z = vector.y
//    }
//}
