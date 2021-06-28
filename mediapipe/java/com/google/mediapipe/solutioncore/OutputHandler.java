// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.solutioncore;

import android.util.Log;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import java.util.List;

/** Interface for handling MediaPipe solution graph outputs. */
public class OutputHandler<T extends SolutionResult> {
  private static final String TAG = "OutputHandler";

  /** Interface for converting outputs packet lists to solution result objects. */
  public interface OutputConverter<T extends SolutionResult> {
    public abstract T convert(List<Packet> packets);
  }
  // A solution specific graph output converter that should be implemented by solution.
  private OutputConverter<T> outputConverter;
  // The user-defined solution result listener.
  private ResultListener<T> customResultListener;
  // The user-defined error listener.
  private ErrorListener customErrorListener;

  /**
   * Sets a callback to be invoked to convert a packet list to a solution result object.
   *
   * @param converter the solution-defined {@link OutputConverter} callback.
   */
  public void setOutputConverter(OutputConverter<T> converter) {
    this.outputConverter = converter;
  }

  /**
   * Sets a callback to be invoked when a solution result objects become available .
   *
   * @param listener the user-defined {@link ResultListener} callback.
   */
  public void setResultListener(ResultListener<T> listener) {
    this.customResultListener = listener;
  }

  /**
   * Sets a callback to be invoked when exceptions are thrown in the solution.
   *
   * @param listener the user-defined {@link ErrorListener} callback.
   */
  public void setErrorListener(ErrorListener listener) {
    this.customErrorListener = listener;
  }

  /** Handles a list of output packets. Invoked when packet lists become available. */
  public void run(List<Packet> packets) {
    T solutionResult = null;
    try {
      solutionResult = outputConverter.convert(packets);
      customResultListener.run(solutionResult);
    } catch (MediaPipeException e) {
      if (customErrorListener != null) {
        customErrorListener.onError("Error occurs when getting MediaPipe solution result. ", e);
      } else {
        Log.e(TAG, "Error occurs when getting MediaPipe solution result. " + e);
      }
    } finally {
      for (Packet packet : packets) {
        packet.release();
      }
      if (solutionResult instanceof ImageSolutionResult) {
        ImageSolutionResult imageSolutionResult = (ImageSolutionResult) solutionResult;
        imageSolutionResult.releaseImagePacket();
      }
    }
  }
}
