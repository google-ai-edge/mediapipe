//
//  ContentViewModel.swift
//  ModelsLabTest
//
//  Created by Mautisim Munir on 12/06/2022.
//

import Foundation
import CoreImage
import UIKit
import SwiftUI
import MPPoseTracking




class ContentViewModel: ObservableObject {
  // 1
  @Published var frame: CGImage?
  // 2
  private let frameManager = FrameManager.shared
    var counter = 0
    
    
    
    
//    let modelPath = Bundle.main.path(forResource: "model", ofType: "edgem")!

//    let model:EdgeModel

  init() {
//      model = EdgeModel(modelPath: modelPath)
    setupSubscriptions()
  }
  // 3
  func setupSubscriptions() {
      // 1
      frameManager.$current
        // 2
        .receive(on: RunLoop.main)
        // 3
        
        
        .compactMap{
            buffer in
            if buffer != nil {
                let ciContext = CIContext()
                let ciImage = CIImage(cvImageBuffer: buffer!)
                let cgImage =  ciContext.createCGImage(ciImage, from: ciImage.extent)

                return cgImage;
            }
            return nil
           
            
        }
        .assign(to: &$frame)

  }
}
