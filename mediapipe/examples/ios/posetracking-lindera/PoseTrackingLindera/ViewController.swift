//
//  ViewController.swift
//  PoseTrackingLindera
//
//  Created by Mautisim Munir on 17/10/2022.
//
import UIKit
import LinderaDetection


class ViewController: UIViewController {

    @IBOutlet  var liveView : UIView!
    @IBOutlet var showLandmarksButton: UIButton!
    @IBOutlet var chooseModelButton: UIButton!
    @IBOutlet var titleview: UIView!
    
    func updateLandmarksButtonText(){
        if (lindera.areLandmarksShown()){
            showLandmarksButton.setTitle("LANDMARKS (ON)", for: UIControl.State.normal)
        }else{
            showLandmarksButton.setTitle("LANDMARKS (OFF)", for: UIControl.State.normal)
        }
        
    }
    
    func updateModelButtonText(){
        
    }
    
    @IBAction func showLandmarksButtonTouch(sender: UIButton){
        
        lindera.showLandmarks(value:  !lindera.areLandmarksShown());
        updateLandmarksButtonText()

//        let alert = UIAlertController(
//            title: nil,
//            message: nil,
//            preferredStyle: .actionSheet
//        )
//
//        alert.addAction(
//            .init(title: "Action 1", style: .default) { _ in
//                print("Action1")
//            }
//        )
//
//        alert.addAction(
//            .init(title: "Action 2", style: .default) { _ in
//                print("Action 2")
//            }
//        )
//
//        present(alert, animated: true)
        

    }
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
        
//        // Do any additional setup after loading the view.
//
//
        self.lindera.delegate = linderaDelegate

        // add lindera camera view to our app's UIView i.e. liveView

        // Expand our cameraView frame to liveView frame
        if let view = self.liveView{
            view.addSubview(lindera.cameraView)
            self.lindera.cameraView.frame = view.bounds

            self.lindera.cameraView.translatesAutoresizingMaskIntoConstraints = false
             NSLayoutConstraint.activate([
                 self.lindera.cameraView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                 self.lindera.cameraView.topAnchor.constraint(equalTo: view.topAnchor),
                 self.lindera.cameraView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
                 self.lindera.cameraView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
             ])
        }


        lindera.startCamera()
        
        self.liveView.bringSubviewToFront(titleview)
        updateLandmarksButtonText()
//        self.liveView.bringSubviewToFront(chooseModelButton)

    }


}

