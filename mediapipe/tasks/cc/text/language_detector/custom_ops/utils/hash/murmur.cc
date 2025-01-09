/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Forked from a library written by Austin Appelby and Jyrki Alakuijala.
// Original copyright message below.
// Copyright 2009 Google Inc.
// Author: aappleby@google.com (Austin Appleby)
//         jyrki@google.com (Jyrki Alakuijala)

#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/hash/murmur.h"

#include <cstdint>

#include "absl/base/internal/endian.h"
#include "absl/base/optimization.h"

namespace mediapipe::tasks::text::language_detector::custom_ops::hash {

namespace {

using ::absl::little_endian::Load64;

// Murmur 2.0 multiplication constant.
static const uint64_t kMul = 0xc6a4a7935bd1e995ULL;

// We need to mix some of the bits that get propagated and mixed into the
// high bits by multiplication back into the low bits. 17 last bits get
// a more efficiently mixed with this.
inline uint64_t ShiftMix(uint64_t val) { return val ^ (val >> 47); }

// Accumulate 8 bytes into 64-bit Murmur hash
inline uint64_t MurmurStep(uint64_t hash, uint64_t data) {
  hash ^= ShiftMix(data * kMul) * kMul;
  hash *= kMul;
  return hash;
}

// Build a uint64_t from 1-8 bytes.
// 8 * len least significant bits are loaded from the memory with
// LittleEndian order. The 64 - 8 * len most significant bits are
// set all to 0.
// In latex-friendly words, this function returns:
//     $\sum_{i=0}^{len-1} p[i] 256^{i}$, where p[i] is unsigned.
//
// This function is equivalent to:
// uint64_t val = 0;
// memcpy(&val, p, len);
// return ToHost64(val);
//
// The caller needs to guarantee that 0 <= len <= 8.
uint64_t Load64VariableLength(const void* const p, int len) {
  ABSL_ASSUME(len >= 0 && len <= 8);
  uint64_t val = 0;
  const uint8_t* const src = static_cast<const uint8_t*>(p);
  for (int i = 0; i < len; ++i) {
    val |= static_cast<uint64_t>(src[i]) << (8 * i);
  }
  return val;
}

}  // namespace

unsigned long long MurmurHash64WithSeed(const char* buf,  // NOLINT
                                        const size_t len, const uint64_t seed) {
  // Let's remove the bytes not divisible by the sizeof(uint64_t).
  // This allows the inner loop to process the data as 64 bit integers.
  const size_t len_aligned = len & ~0x7;
  const char* const end = buf + len_aligned;
  uint64_t hash = seed ^ (len * kMul);
  for (const char* p = buf; p != end; p += 8) {
    hash = MurmurStep(hash, Load64(p));
  }
  if ((len & 0x7) != 0) {
    const uint64_t data = Load64VariableLength(end, len & 0x7);
    hash ^= data;
    hash *= kMul;
  }
  hash = ShiftMix(hash) * kMul;
  hash = ShiftMix(hash);
  return hash;
}

}  // namespace mediapipe::tasks::text::language_detector::custom_ops::hash
