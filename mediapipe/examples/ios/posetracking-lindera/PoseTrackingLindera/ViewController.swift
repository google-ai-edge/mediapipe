//
//  ViewController.swift
//  PoseTrackingLindera
//
//  Created by Mautisim Munir on 17/10/2022.
//

import UIKit
import MPPoseTracking

class ViewController: UIViewController {
    let poseTracking:PoseTracking = PoseTracking(poseTrackingOptions: PoseTrackingOptions(showLandmarks: true));
    let cameraSource = MPPCameraInputSource();
    @IBOutlet  var liveView:UIView?;

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
       
        self.poseTracking.renderer.layer.frame = self.liveView!.layer.bounds
        self.liveView?.layer.addSublayer(self.poseTracking.renderer.layer)
        self.cameraSource.sessionPreset = AVCaptureSession.Preset.high.rawValue;
        self.cameraSource.cameraPosition = AVCaptureDevice.Position.front;
        self.cameraSource.orientation = AVCaptureVideoOrientation.portrait;
        if (self.cameraSource.orientation == AVCaptureVideoOrientation.portrait){
            self.cameraSource.videoMirrored = true;
        }
        self.cameraSource.requestCameraAccess(
            completionHandler: {(granted:Bool)->Void
            in
                if (granted){
                    self.poseTracking.start(withCamera: self.cameraSource)
                }
            
        })
    }


}

