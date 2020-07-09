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

package com.google.mediapipe.components;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.util.Log;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

/** Manages camera permission request and handling. */
public class PermissionHelper {
  private static final String TAG = "PermissionHelper";

  private static final String AUDIO_PERMISSION = Manifest.permission.RECORD_AUDIO;

  private static final String CAMERA_PERMISSION = Manifest.permission.CAMERA;

  private static final int REQUEST_CODE = 0;

  public static boolean permissionsGranted(Activity context, String[] permissions) {
    for (String permission : permissions) {
      int permissionStatus = ContextCompat.checkSelfPermission(context, permission);
      if (permissionStatus != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  public static void checkAndRequestPermissions(Activity context, String[] permissions) {
    if (!permissionsGranted(context, permissions)) {
      ActivityCompat.requestPermissions(context, permissions, REQUEST_CODE);
    }
  }

  /** Called by context to check if camera permissions have been granted. */
  public static boolean cameraPermissionsGranted(Activity context) {
    return permissionsGranted(context, new String[] {CAMERA_PERMISSION});
  }

  /**
   * Called by context to check if camera permissions have been granted and if not, request them.
   */
  public static void checkAndRequestCameraPermissions(Activity context) {
    Log.d(TAG, "checkAndRequestCameraPermissions");
    checkAndRequestPermissions(context, new String[] {CAMERA_PERMISSION});
  }

  /** Called by context to check if audio permissions have been granted. */
  public static boolean audioPermissionsGranted(Activity context) {
    return permissionsGranted(context, new String[] {AUDIO_PERMISSION});
  }

  /** Called by context to check if audio permissions have been granted and if not, request them. */
  public static void checkAndRequestAudioPermissions(Activity context) {
    Log.d(TAG, "checkAndRequestAudioPermissions");
    checkAndRequestPermissions(context, new String[] {AUDIO_PERMISSION});
  }

  /** Called by context when permissions request has been completed. */
  public static void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    Log.d(TAG, "onRequestPermissionsResult");
    if (permissions.length > 0 && grantResults.length != permissions.length) {
      Log.d(TAG, "Permission denied.");
      return;
    }
    for (int i = 0; i < grantResults.length; ++i) {
      if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
        Log.d(TAG, permissions[i] + " permission granted.");
      }
    }
    // Note: We don't need any special callbacks when permissions are ready because activities
    // using this helper class can have code in onResume() which is called after the
    // permissions dialog box closes. The code can be branched depending on if permissions are
    // available via permissionsGranted(Activity).
    return;
  }
}
