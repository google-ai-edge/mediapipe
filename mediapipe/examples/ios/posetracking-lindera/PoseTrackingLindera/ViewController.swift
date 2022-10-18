//
//  ViewController.swift
//  PoseTrackingLindera
//
//  Created by Mautisim Munir on 17/10/2022.
//
import UIKit
import LinderaDetection


class ViewController: UIViewController {

    @IBOutlet  var liveView:UIView?;
    
    let lindera =  Lindera()
    
    /// A simple LinderaDelegate implementation that prints nose coordinates if detected
    class LinderaDelegateImpl:LinderaDelegate{
        func lindera(_ lindera: Lindera, didDetect event: Asensei3DPose.Event) {
            if let kpt = event.pose.nose{
                print("LinderaDelegateImpl: Nose Keypoint (\(String(describing: kpt.position.x)),\(String(describing: kpt.position.y)),\(kpt.position.z)) with confidence \(kpt.confidence)")
            }
        }
        
        
    }

    let linderaDelegate = LinderaDelegateImpl()
    

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Do any additional setup after loading the view.
        

        self.lindera.delegate = linderaDelegate
        
        // add lindera camera view to our app's UIView i.e. liveView
        self.liveView?.addSubview(lindera.cameraView)
        // Expand our cameraView frame to liveView frame
        lindera.cameraView.frame = self.liveView!.bounds;
        
        lindera.startCamera()

    }


}

