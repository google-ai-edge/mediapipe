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

import ARKit
import AVFoundation

class FrontCamera: NSObject {
  lazy var session: AVCaptureSession = .init()

  lazy var device: AVCaptureDevice = AVCaptureDevice.default(
    .builtInWideAngleCamera, for: .video, position: .front)!

  lazy var input: AVCaptureDeviceInput = try! AVCaptureDeviceInput(device: device)
  lazy var output: AVCaptureVideoDataOutput = .init()

  override init() {
    super.init()
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    session.addInput(input)
    session.addOutput(output)
  }

  func setSampleBufferDelegate(_ delegate: AVCaptureVideoDataOutputSampleBufferDelegate) {
    output.setSampleBufferDelegate(delegate, queue: .main)
  }

  func start() {
    session.startRunning()
  }

  func stop() {
    session.stopRunning()
  }
}
