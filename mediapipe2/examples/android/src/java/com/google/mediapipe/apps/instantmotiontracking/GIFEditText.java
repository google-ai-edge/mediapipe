// Copyright 2020 Google LLC
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

package com.google.mediapipe.apps.instantmotiontracking;

import android.content.ClipDescription;
import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import androidx.appcompat.widget.AppCompatEditText;
import android.util.AttributeSet;
import android.util.Log;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import androidx.core.view.inputmethod.EditorInfoCompat;
import androidx.core.view.inputmethod.InputConnectionCompat;
import androidx.core.view.inputmethod.InputContentInfoCompat;

// import android.support.v13.view.inputmethod.EditorInfoCompat;
// import android.support.v13.view.inputmethod.InputConnectionCompat;
// import android.support.v13.view.inputmethod.InputContentInfoCompat;

/**
 * This custom EditText implementation uses the existing EditText framework in
 * order to develop a GIFEditText input box which is capable of accepting GIF
 * animations from the Android system keyboard and return the GIF location with
 * a content URI.
 */
public class GIFEditText extends AppCompatEditText {

  private GIFCommitListener gifCommitListener;

  public GIFEditText(Context context) {
    super(context);
  }

  public GIFEditText(Context context, AttributeSet attrs) {
    super(context, attrs);
  }

  /**
   * onGIFCommit is called once content is pushed to the EditText via the
   * Android keyboard.
   */
  public interface GIFCommitListener {
    void onGIFCommit(Uri contentUri, ClipDescription description);
  }

  /**
   * Used to set the gifCommitListener for this GIFEditText.
   *
   * @param gifCommitListener handles response to new content pushed to EditText
   */
  public void setGIFCommitListener(GIFCommitListener gifCommitListener) {
    this.gifCommitListener = gifCommitListener;
  }

  @Override
  public InputConnection onCreateInputConnection(EditorInfo editorInfo) {
    final InputConnection inputConnection = super.onCreateInputConnection(editorInfo);
    EditorInfoCompat.setContentMimeTypes(editorInfo, new String[] {"image/gif"});
    return InputConnectionCompat.createWrapper(
        inputConnection,
        editorInfo,
        new InputConnectionCompat.OnCommitContentListener() {
          @Override
          public boolean onCommitContent(
              final InputContentInfoCompat inputContentInfo, int flags, Bundle opts) {
            try {
              if (gifCommitListener != null) {
                Runnable runnable =
                    new Runnable() {
                      @Override
                      public void run() {
                        inputContentInfo.requestPermission();
                        gifCommitListener.onGIFCommit(
                            inputContentInfo.getContentUri(), inputContentInfo.getDescription());
                        inputContentInfo.releasePermission();
                      }
                    };
                new Thread(runnable).start();
              }
            } catch (RuntimeException e) {
              Log.e("GIFEditText", "Input connection to GIF selection failed");
              e.printStackTrace();
              return false;
            }
            return true;
          }
        });
  }
}
