//
//  utils.swift
//  Mediapipe
//
//  Created by Mautisim Munir on 21/10/2022.
//

import Foundation


public class FPSHelper{
    var smoothingFactor = 0.8
    var _fps:Double? = nil
    var time: CFAbsoluteTime? = nil
    public var onFpsUpdate : ((_ fps:Double)->Void)? = nil
    init(smoothingFactor:Double) {
        self.smoothingFactor = smoothingFactor
    }
    
    public func logTime(){
        
        let currTime = CFAbsoluteTimeGetCurrent()
        if (time != nil){
        
            let elapsedTime = currTime - time!
            let fps = 1/Double(elapsedTime)
            if (_fps == nil){
                _fps = fps
            }else{
                _fps = (1-smoothingFactor)*fps + smoothingFactor*_fps!
            }
            if (onFpsUpdate != nil){
                onFpsUpdate?(_fps!)
            }
            
        }
        time = currTime
        
    }
    
    
    
    
    
}
