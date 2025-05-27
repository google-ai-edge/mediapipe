// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_API3_PACKET_H_
#define MEDIAPIPE_FRAMEWORK_API3_PACKET_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe::api3 {

// A generic container class which can hold data of a specific type.
//
// `Packet` is implemented as a reference-counted pointer.  This means
// that copying `Packet`s creates a fast, shallow copy.  `Packet`s are
// copyable, movable, and assignable.  `Packet`s can be stored in STL
// containers.  A `Packet` may optionally contain a timestamp.
//
// The preferred method of creating a `Packet` is with `MakePacket<T>()`.
// `Packet` typically owns the object that it contains, but `PointToForeign`
// allows a Packet to be constructed which does not own its data.
//
// This class is thread compatible.
template <typename T>
class Packet {
 public:
  explicit Packet() = default;
  explicit Packet(mediapipe::Packet p) : packet_(std::move(p)) {}

  Packet(const Packet& p) = default;
  Packet& operator=(const Packet& p) = default;

  Packet(Packet&& p) = default;
  Packet& operator=(Packet&& p) = default;

  explicit operator bool() const { return !packet_.IsEmpty(); }

  const T& GetOrDie() const { return packet_.Get<T>(); }

  Packet<T> At(Timestamp timestamp) const {
    return Packet(packet_.At(timestamp));
  }

  mediapipe::Timestamp Timestamp() const { return packet_.Timestamp(); }

  const mediapipe::Packet& AsLegacyPacket() const { return packet_; }

  std::string DebugString() const { return packet_.DebugString(); }

 private:
  mediapipe::Packet packet_;
};

// Create a packet containing an object of type T initialized with the
// provided arguments.
//
// The timestamp of the returned Packet is Timestamp::Unset(). To set the
// timestamp, the caller should do PointToForeign(...).At(...).
template <typename T, typename... Args>
Packet<T> MakePacket(Args&&... args) {
  return Packet<T>(mediapipe::MakePacket<T>(std::forward<Args>(args)...));
}
template <typename T>
Packet<T> MakePacket(std::unique_ptr<T> ptr) {
  return Packet<T>(mediapipe::Adopt(ptr.release()));
}

// Returns a Packet that does not own its data. The data pointed to by *ptr
// remains owned by the caller, who must ensure that it outlives not only the
// returned Packet but also all of its copies.
//
// Optionally, `cleanup` object can be specified to invoke when all copies of
// the packet are destroyed (can be used to capture the foreign owner if
// possible and ensure the lifetime).
//
// The timestamp of the returned Packet is Timestamp::Unset(). To set the
// timestamp, the caller should do PointToForeign(...).At(...).
template <typename T>
Packet<T> PointToForeign(const T* ptr,
                         absl::AnyInvocable<void()> cleanup = nullptr) {
  return Packet<T>(mediapipe::PointToForeign(ptr, std::move(cleanup)));
}

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_PACKET_H_
