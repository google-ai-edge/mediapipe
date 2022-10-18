import Foundation

public struct Asensei3DPose {

    
    public let nose: BodyJointDetails?
       
    public let leftEyeInner: BodyJointDetails?
    public let leftEye: BodyJointDetails?
    public let leftEyeOuter: BodyJointDetails?

    public let rightEyeInner: BodyJointDetails?
    public let rightEye: BodyJointDetails?
    public let rightEyeOuter: BodyJointDetails?
       
    public let leftEar: BodyJointDetails?
    public let rightEar: BodyJointDetails?

    public let mouthLeft: BodyJointDetails?
    public let mouthRight: BodyJointDetails?

    public let leftShoulder: BodyJointDetails?
    public let rightShoulder: BodyJointDetails?

    public let leftElbow: BodyJointDetails?
    public let rightElbow: BodyJointDetails?

    public let leftWrist: BodyJointDetails?
    public let rightWrist: BodyJointDetails?

    public let leftPinky: BodyJointDetails?
    public let rightPinky: BodyJointDetails?

    public let leftIndex: BodyJointDetails?
    public let rightIndex: BodyJointDetails?

    public let leftThumb: BodyJointDetails?
    public let rightThumb: BodyJointDetails?

    public let leftHip: BodyJointDetails?
    public let rightHip: BodyJointDetails?

    public let leftKnee: BodyJointDetails?
    public let rightKnee: BodyJointDetails?
       
    public let rightAnkle: BodyJointDetails?
    public let leftAnkle: BodyJointDetails?


    public let rightHeel: BodyJointDetails?
    public let leftHeel: BodyJointDetails?

    public let rightFoot: BodyJointDetails?
    public let leftFoot: BodyJointDetails?
}

extension Asensei3DPose: Encodable {

    private enum CodingKeys: String, CodingKey {
            case nose
           
           case leftEyeInner
           case leftEye
           case leftEyeOuter

           case rightEyeInner
           case rightEye
           case rightEyeOuter
           
           case leftEar
           case rightEar

           case mouthLeft
           case mouthRight

           case leftShoulder
           case rightShoulder

           case leftElbow
           case rightElbow

           case leftWrist
           case rightWrist

           case leftPinky
           case rightPinky

           case leftIndex
           case rightIndex

           case leftThumb
           case rightThumb

           case leftHip
           case rightHip

           case leftKnee
           case rightKnee
           
           case rightAnkle
           case leftAnkle


           case rightHeel
           case leftHeel

           case rightFoot
           case leftFoot
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encodeIfPresent(self.nose, forKey: .nose)
           
        try container.encodeIfPresent(self.leftEyeInner, forKey: .leftEyeInner)
        try container.encodeIfPresent(self.leftEye, forKey:.leftEye )
        try container.encodeIfPresent(self.leftEyeOuter, forKey: .leftEyeOuter)

        try container.encodeIfPresent(self.rightEyeInner, forKey: .rightEyeInner)
        try container.encodeIfPresent(self.rightEye, forKey: .rightEye)
        try container.encodeIfPresent(self.rightEyeOuter, forKey: .rightEyeOuter )
           
        try container.encodeIfPresent(self.leftEar, forKey: .leftEar)
        try container.encodeIfPresent(self.rightEar, forKey: .rightEar)

        try container.encodeIfPresent(self.mouthLeft, forKey: .mouthLeft)
        try container.encodeIfPresent(self.mouthRight, forKey: .mouthRight )

        try container.encodeIfPresent(self.leftShoulder, forKey: .leftShoulder)
        try container.encodeIfPresent(self.rightShoulder, forKey: .rightShoulder)

        try container.encodeIfPresent(self.leftElbow, forKey: .leftElbow)
        try container.encodeIfPresent(self.rightElbow, forKey:.rightElbow )

        try container.encodeIfPresent(self.leftWrist, forKey: .leftWrist)
        try container.encodeIfPresent(self.rightWrist, forKey: .rightWrist )

        try container.encodeIfPresent(self.leftPinky, forKey: .leftPinky)
        try container.encodeIfPresent(self.rightPinky, forKey: .rightPinky)

        try container.encodeIfPresent(self.leftIndex, forKey: .leftIndex )
        try container.encodeIfPresent(self.rightIndex, forKey:.rightIndex )

        try container.encodeIfPresent(self.leftThumb, forKey: .leftThumb)
        try container.encodeIfPresent(self.rightThumb, forKey: .rightThumb )

        try container.encodeIfPresent(self.leftHip, forKey: .leftHip)
        try container.encodeIfPresent(self.rightHip, forKey: .rightHip )

        try container.encodeIfPresent(self.leftKnee, forKey: .leftKnee )
        try container.encodeIfPresent(self.rightKnee, forKey: .rightKnee )
           
        try container.encodeIfPresent(self.rightAnkle, forKey: .rightAnkle)
        try container.encodeIfPresent(self.leftAnkle, forKey: .leftAnkle )


        try container.encodeIfPresent(self.rightHeel, forKey: .rightHeel)
        try container.encodeIfPresent(self.leftHeel, forKey: .leftHeel)

        try container.encodeIfPresent(self.rightFoot, forKey: .rightFoot )
        try container.encodeIfPresent(self.leftFoot, forKey: .leftFoot)
        
  
    }
}

extension Asensei3DPose {

    public struct BodyJointDetails: Encodable {

        public let position: Vector3D
        public let confidence: Float

        private enum CodingKeys: String, CodingKey {
            case x
            case y
            case z
            case c
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(self.position.x, forKey: .x)
            try container.encode(self.position.y, forKey: .y)
            try container.encode(self.position.z, forKey: .z)
            try container.encode(self.confidence, forKey: .c)
        }
    }
}

extension Asensei3DPose {

    public struct Vector3D {
        public let x: Float
        public let y: Float
        public let z: Float

        public init(x: Float, y: Float, z: Float) {
            self.x = x
            self.y = y
            self.z = z
        }
    }
}

extension Asensei3DPose {

    public struct Event: Encodable {
        public let pose: Asensei3DPose
        let timestamp: TimeInterval

        private enum CodingKeys: String, CodingKey {
            case bodyJoints
            case timestamp
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(self.pose, forKey: .bodyJoints)
            try container.encode(self.timestamp * 1000, forKey: .timestamp)
        }
    }
}
