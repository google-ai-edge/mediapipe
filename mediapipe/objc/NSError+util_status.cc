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

#import "mediapipe/objc/NSError+util_status.h"

@implementation GUSUtilStatusWrapper

+ (instancetype)wrapStatus:(const absl::Status &)status {
  return [[self alloc] initWithStatus:status];
}

- (instancetype)initWithStatus:(const absl::Status &)status {
  self = [super init];
  if (self) {
    _status = status;
  }
  return self;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"<%@: %p; status = %s>", [self class],
                                    self, _status.message().data()];
}

@end

@implementation NSError (GUSGoogleUtilStatus)

NSString *const kGUSGoogleUtilStatusErrorDomain =
    @"GoogleUtilStatusErrorDomain";
NSString *const kGUSGoogleUtilStatusErrorKey = @"GUSGoogleUtilStatusErrorKey";

+ (NSError *)gus_errorWithStatus:(const absl::Status &)status {
  NSDictionary *userInfo = @{
    NSLocalizedDescriptionKey : @(status.message().data()),
    kGUSGoogleUtilStatusErrorKey : [GUSUtilStatusWrapper wrapStatus:status],
  };
  NSError *error =
      [NSError errorWithDomain:kGUSGoogleUtilStatusErrorDomain
                          code:static_cast<NSInteger>(status.code())
                      userInfo:userInfo];
  return error;
}

- (absl::Status)gus_status {
  NSString *domain = self.domain;
  if ([domain isEqual:kGUSGoogleUtilStatusErrorDomain]) {
    GUSUtilStatusWrapper *wrapper = self.userInfo[kGUSGoogleUtilStatusErrorKey];
    if (wrapper) return wrapper.status;
#if 0
  // Unfortunately, util/task/posixerrorspace.h is not in portable status yet.
  // TODO: fix that.
  } else if ([domain isEqual:NSPOSIXErrorDomain]) {
    return ::util::PosixErrorToStatus(self.code, self.localizedDescription.UTF8String);
#endif
  }
  return absl::Status(absl::StatusCode::kUnknown,
                      self.localizedDescription.UTF8String);
}

@end
