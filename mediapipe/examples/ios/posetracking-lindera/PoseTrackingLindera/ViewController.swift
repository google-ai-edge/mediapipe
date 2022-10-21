//
//  ViewController.swift
//  PoseTrackingLindera
//
//  Created by Mautisim Munir on 17/10/2022.
//
import UIKit
import LinderaDetection


class ViewController: UIViewController {
    
    //MARK: - UI Elements
    
    
    @IBOutlet  var liveView : UIView!
    @IBOutlet var showLandmarksButton: UIButton!
    @IBOutlet var chooseModelButton: UIButton!
    @IBOutlet var titleview: UIView!
    @IBOutlet var fpsLabel: UILabel!
    
    
    //MARK: - UI Actions
    
    @IBAction func setModelComplexity(){
        let alert = UIAlertController(
            title: nil,
            message: nil,
            preferredStyle: .actionSheet
        )
        
        alert.addAction(
            .init(title: "MODEL (LITE)", style: .default) {[weak self] _ in
                self?.lindera.setModelComplexityNow(complexity: 0)
                self?.updateModelButtonText()
                
            }
        )
        
        alert.addAction(
            .init(title: "MODEL (FULL)", style: .default) { [weak self] _ in
                self?.lindera.setModelComplexityNow(complexity: 1)
                self?.updateModelButtonText()
                
                
            }
        )
        alert.addAction(
            .init(title: "MODEL (HEAVY)", style: .default) { [weak self] _ in
                self?.lindera.setModelComplexityNow(complexity: 2)
                self?.updateModelButtonText()
                
                
            }
        )
        
        present(alert, animated: true)
    }
    
    @IBAction func showLandmarksButtonTouch(sender: UIButton){
        
        lindera.showLandmarks(value:  !lindera.areLandmarksShown());
        updateLandmarksButtonText()
        
    }
    
    // MARK: - LinderaDelegate
    
    /// A simple LinderaDelegate implementation that prints nose coordinates if detected
    class LinderaDelegateImpl:LinderaDelegate{
        func lindera(_ lindera: Lindera, didDetect event: Asensei3DPose.Event) {
            //            if let kpt = event.pose.nose{
            //                // Printing causes large drops in FPS
            //                print("LinderaDelegateImpl: Nose Keypoint (\(String(describing: kpt.position.x)),\(String(describing: kpt.position.y)),\(kpt.position.z)) with confidence \(kpt.confidence)")
            //            }
        }
        
        
    }
    // MARK: - UI Text Modifications
    func updateLandmarksButtonText(){
        if (lindera.areLandmarksShown()){
            showLandmarksButton.setTitle("LANDMARKS (ON)", for: UIControl.State.normal)
        }else{
            showLandmarksButton.setTitle("LANDMARKS (OFF)", for: UIControl.State.normal)
        }
        
    }
    
    func updateModelButtonText(){
        var text = "MODEL "
        switch(lindera.getModelComplexity()){
            
        case 0:
            text += "(LITE)"
            break;
        case 1:
            text += "(FULL)"
            break;
        case 2:
            text += "(HEAVY)"
            break;
            
        default:
            text += "(Unknown)"
        }
        chooseModelButton.setTitle(text, for: UIControl.State.normal)
    }
    
    
    
    // MARK: - State Objects
    
    let lindera =  Lindera()

    let linderaDelegate = LinderaDelegateImpl()
    
    // MARK: - UI Setup
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.lindera.delegate = linderaDelegate
        
        
        if let view = self.liveView{
            // add lindera camera view to our app's UIView i.e. liveView
            view.addSubview(lindera.cameraView)
            // Expand our cameraView frame to liveView frame
            self.lindera.cameraView.frame = view.bounds
        
            // Setting Up Constraints (No necessary with above statement)
            self.lindera.cameraView.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                self.lindera.cameraView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                self.lindera.cameraView.topAnchor.constraint(equalTo: view.topAnchor),
                self.lindera.cameraView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
                self.lindera.cameraView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
            ])
        }
        
        // This function is called whenver there is an fps update
        self.lindera.setFpsDelegate(fpsDelegate: {[weak self] fps in
            DispatchQueue.main.async {
                self?.fpsLabel.text = "\(Int(fps)) fps"
            }
            
        })
        
        // Otherwise they are hidden
        self.liveView.bringSubviewToFront(titleview)
        self.liveView.bringSubviewToFront(fpsLabel)
        
        // Make the Landmarks and Model button text reflect the state in lindera object
        updateLandmarksButtonText()
        updateModelButtonText()
        
        lindera.startCamera()

        
    }
    
    
}

