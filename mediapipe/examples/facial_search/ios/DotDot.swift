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

/**
 Custom operator that calls the specified block `self` value as its argument and returns `self`.

 Usage:

 self.backgroundView = UILabel()..{
 $0.backgroundColor = .red
 $0.textColor = .white
 $0.numberOfLines = 0
 }
 */

infix operator ..: MultiplicationPrecedence

@discardableResult
public func .. <T>(object: T, block: (inout T) -> Void) -> T {
  var object = object
  block(&object)
  return object
}
