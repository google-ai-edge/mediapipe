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

package com.google.mediapipe.framework;

// Package java.nio.charset is not yet available in all Android apps.
import static com.google.common.base.Charsets.UTF_8;

/** This class represents an error reported by the MediaPipe framework. */
public class MediaPipeException extends RuntimeException {
  public MediaPipeException(int statusCode, String statusMessage) {
    super(StatusCode.values()[statusCode].description() + ": " + statusMessage);
    this.statusCode = StatusCode.values()[statusCode];
    this.statusMessage = statusMessage;
  }

  // Package base.Charsets is deprecated by package java.nio.charset is not
  // yet available in all Android apps.
  @SuppressWarnings("deprecation")
  MediaPipeException(int code, byte[] message) {
    this(code, new String(message, UTF_8));
  }

  public StatusCode getStatusCode() {
    return statusCode;
  }

  public String getStatusMessage() {
    return statusMessage;
  }

  /** The 17 canonical status codes. */
  public enum StatusCode {
    OK("ok"),
    CANCELLED("canceled"),
    UNKNOWN("unknown"),
    INVALID_ARGUMENT("invalid argument"),
    DEADLINE_EXCEEDED("deadline exceeded"),
    NOT_FOUND("not found"),
    ALREADY_EXISTS("already exists"),
    PERMISSION_DENIED("permission denied"),
    RESOURCE_EXHAUSTED("resource exhausted"),
    FAILED_PRECONDITION("failed precondition"),
    ABORTED("aborted"),
    OUT_OF_RANGE("out of range"),
    UNIMPLEMENTED("unimplemented"),
    INTERNAL("internal"),
    UNAVAILABLE("unavailable"),
    DATA_LOSS("data loss"),
    UNAUTHENTICATED("unauthenticated");

    StatusCode(String description) {
      this.description = description;
    }

    public String description() {
      return description;
    }

    private final String description;
  };

  private final StatusCode statusCode;
  private final String statusMessage;
}
