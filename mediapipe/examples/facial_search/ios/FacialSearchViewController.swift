// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import AVFoundation
import SceneKit
import UIKit

#if canImport(FacialSearch)
  // Either import standalone iOS framework...
  import FacialSearch

  func facialSearchBundle() -> Bundle {
    return Bundle(for: FacialSearch.self)
  }
#elseif canImport(mediapipe_examples_facial_search_ios_ObjcppLib)
  // ...or import the ObjcppLib target linked using Bazel.
  import mediapipe_examples_facial_search_ios_ObjcppLib

  func facialSearchBundle() -> Bundle {
    return Bundle.main
  }
#endif

class FacialSearchViewController: UIViewController,
  AVCaptureVideoDataOutputSampleBufferDelegate, FacialSearchDelegate
{
  let camera = FrontCamera()
  let displayLayer: AVSampleBufferDisplayLayer = .init()
  let ffind: FacialSearch = FacialSearch()!

  private lazy var cameraView: UIView = UIView() .. {
    $0.translatesAutoresizingMaskIntoConstraints = false
  }

  private lazy var containView: UIView = UIView() .. {
    $0.translatesAutoresizingMaskIntoConstraints = false
    $0.clipsToBounds = true
  }

  private lazy var imgView: UIImageView = UIImageView() .. {
    $0.translatesAutoresizingMaskIntoConstraints = false
    $0.contentMode = .scaleAspectFill
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    view.addSubview(cameraView)
    view.addSubview(containView)
    containView.addSubview(imgView)

    NSLayoutConstraint.activate([
      cameraView.topAnchor.constraint(equalTo: view.topAnchor),
      cameraView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      cameraView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      cameraView.heightAnchor.constraint(equalTo: view.widthAnchor),
      containView.topAnchor.constraint(equalTo: cameraView.bottomAnchor),
      containView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      containView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      containView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
      imgView.topAnchor.constraint(equalTo: containView.topAnchor),
      imgView.leadingAnchor.constraint(equalTo: containView.leadingAnchor),
      imgView.trailingAnchor.constraint(equalTo: containView.trailingAnchor),
      imgView.bottomAnchor.constraint(equalTo: containView.bottomAnchor),
    ])

    cameraView.layer.addSublayer(displayLayer)

    camera.setSampleBufferDelegate(self)
    camera.start()
    ffind.startGraph()
    ffind.delegate = self
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    displayLayer.frame = cameraView.bounds
  }

  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    connection.videoOrientation = AVCaptureVideoOrientation.portrait

    displayLayer.enqueue(sampleBuffer)
    let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
    ffind.processVideoFrame(pixelBuffer)
  }

  func didReceive(_ memes: [Classification]!) {
    NSLog("Got %d memes!", memes.count)
    if memes.count != 0 {
      DispatchQueue.main.async {
        if let img = self.getImage(name: memes.last!.label) {
          self.imgView.contentMode = .scaleAspectFill
          self.imgView.image = img
        }
      }
    }
  }

  func getImage(name: String) -> UIImage? {
    NSLog("Loading image %@", name)
    let nameWE = NSString(string: name).deletingPathExtension as String
    let bundle = facialSearchBundle()
    guard let url = bundle.url(forResource: nameWE, withExtension: "jpg") else {
      fatalError("Can't load image")
    }
    NSLog("Image path: %@", url.path)
    guard let image = UIImage(contentsOfFile: url.path) else {
      fatalError("Image is not a .jpg")
    }
    NSLog("Image size: %@", image.size.debugDescription)
    return image
  }

  func resizeImage(img: UIImage, size: CGSize) -> UIImage? {
    let renderer = UIGraphicsImageRenderer(size: size)
    return renderer.image { (context) in
      img.draw(in: CGRect(origin: .zero, size: size))
    }
  }
}
