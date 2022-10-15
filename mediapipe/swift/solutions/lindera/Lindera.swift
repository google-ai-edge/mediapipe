final class Lindera {

    var cameraView: UIView {
        self.linderaExerciseSession
    }

    weak var delegate: LinderaDelegate?

    private lazy var linderaExerciseSession: LinderaExerciseSessionView = {
        let session = LinderaExerciseSessionView()
        session.detectionSpeed = .high
        session.processCameraFrames = true
        session.enable3DPoseDetection = true
        session.pose3DDetectionHandler = { [weak self] event in

            guard let self = self, let pose = event.pose.map({ Asensei3DPose($0) }) else {
                return
            }

            self.delegate?.lindera(self, didDetect: .init(pose: pose, timestamp: event.sourceTimestamp))
        }

        return session
    }()

    required init () { }

    func startCamera(_ completion: ((Result<Void, Error>) -> Void)? = nil) {
        DispatchQueue.main.async { [weak self] in
            self?.linderaExerciseSession.startCamera { result in
                switch result {
                case .success:
                    completion?(.success(Void()))
                case .failure(let error):
                    completion?(.failure(error))
                }
            }
        }
    }

    func stopCamera() {
        DispatchQueue.main.async { [weak self] in
            self?.linderaExerciseSession.stopCamera()
        }
    }

    func switchCamera(_ completion: ((Result<Void, Error>) -> Void)? = nil) {
        DispatchQueue.main.async { [weak self] in
            self?.linderaExerciseSession.switchCamera(completionHandler: completion)
        }
    }

    func selectCamera(_ position: AVCaptureDevice.Position, _ completion: ((Result<Void, Error>) -> Void)? = nil) {
        DispatchQueue.main.async { [weak self] in
            self?.linderaExerciseSession.setUseFrontCamera(position == .front, completionHandler: completion)
        }
    }
}

protocol LinderaDelegate: AnyObject {

    func lindera(_ lindera: Lindera, didDetect event: Asensei3DPose.Event)
}

// MARK: - Helpers
extension Asensei3DPose {

    init(_ pose: Lindera3DPose) {
        self.pelvis = pose.landmarks[.pelvis].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.rightHip = pose.landmarks[.rightHip].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.rightKnee = pose.landmarks[.rightKnee].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.rightAnkle = pose.landmarks[.rightAnkle].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.leftHip = pose.landmarks[.leftHip].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.leftKnee = pose.landmarks[.leftKnee].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.leftAnkle = pose.landmarks[.leftAnkle].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.spine = pose.landmarks[.spine].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.thorax = pose.landmarks[.thorax].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.neckNose = pose.landmarks[.neckToNose].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.headTop = pose.landmarks[.headTop].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.leftShoulder = pose.landmarks[.leftShoulder].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.leftElbow = pose.landmarks[.leftElbow].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.leftWrist = pose.landmarks[.leftWrist].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.rightShoulder = pose.landmarks[.rightShoulder].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.rightElbow = pose.landmarks[.rightElbow].map { .init(position: .init($0.position), confidence: $0.confidence) }
        self.rightWrist = pose.landmarks[.rightWrist].map { .init(position: .init($0.position), confidence: $0.confidence) }
    }
}

extension Asensei3DPose.Vector3D {

    init(_ vector: Lindera3DVector) {
        self.x = -vector.x
        self.y = vector.z
        self.z = vector.y
    }
}
