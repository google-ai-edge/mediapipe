import SwiftUI
import MPPoseTracking





struct ContentView: View {
    @StateObject private var model = ContentViewModel()
    
    let poseTracking = PoseTracking(poseTrackingOptions: PoseTrackingOptions(showLandmarks: true))
    

    
    init() {
        poseTracking?.renderer.layer.frame = self.body.layer
        
    }
    
    var body: some View {
            VStack{
            FrameView(image: model.frame)
              .edgesIgnoringSafeArea(.all)
//            buildInferenceView()
            }
        

    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
