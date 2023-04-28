// Copyright 2022 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.examples.objectdetector;

import android.content.Intent;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.os.Bundle;
import android.provider.MediaStore;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.exifinterface.media.ExifInterface;
// ContentResolver dependency
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectionResult;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector.ObjectDetectorOptions;
import java.io.IOException;
import java.io.InputStream;

/** Main activity of MediaPipe Task Object Detector reference app. */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";
  private static final String MODEL_FILE = "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite";

  private ObjectDetector objectDetector;

  private enum InputSource {
    UNKNOWN,
    IMAGE,
    VIDEO,
    CAMERA,
  }

  private InputSource inputSource = InputSource.UNKNOWN;

  // Image mode demo component.
  private ActivityResultLauncher<Intent> imageGetter;
  // Video mode demo component.
  private ActivityResultLauncher<Intent> videoGetter;
  private ObjectDetectionResultImageView imageView;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    setupImageModeDemo();
    setupVideoModeDemo();
    // TODO: Adds live camera demo.
  }

  /** Sets up the image mode demo. */
  private void setupImageModeDemo() {
    imageView = new ObjectDetectionResultImageView(this);
    // The Intent to access gallery and read images as bitmap.
    imageGetter =
        registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
              Intent resultIntent = result.getData();
              if (resultIntent != null) {
                if (result.getResultCode() == RESULT_OK) {
                  Bitmap bitmap = null;
                  int rotation = 0;
                  try {
                    bitmap =
                        downscaleBitmap(
                            MediaStore.Images.Media.getBitmap(
                                this.getContentResolver(), resultIntent.getData()));
                  } catch (IOException e) {
                    Log.e(TAG, "Bitmap reading error:" + e);
                  }
                  try {
                    InputStream imageData =
                        this.getContentResolver().openInputStream(resultIntent.getData());
                    rotation = getImageRotation(imageData);
                  } catch (IOException | MediaPipeException e) {
                    Log.e(TAG, "Bitmap rotation error:" + e);
                  }
                  if (bitmap != null) {
                    MPImage image = new BitmapImageBuilder(bitmap).build();
                    ObjectDetectionResult detectionResult =
                        objectDetector.detect(
                            image,
                            ImageProcessingOptions.builder().setRotationDegrees(rotation).build());
                    imageView.setData(image, detectionResult);
                    runOnUiThread(() -> imageView.update());
                  }
                }
              }
            });
    Button loadImageButton = findViewById(R.id.button_load_picture);
    loadImageButton.setOnClickListener(
        v -> {
          if (inputSource != InputSource.IMAGE) {
            createObjectDetector(RunningMode.IMAGE);
            this.inputSource = InputSource.IMAGE;
            updateLayout();
          }
          // Reads images from gallery.
          Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
          pickImageIntent.setDataAndType(MediaStore.Images.Media.INTERNAL_CONTENT_URI, "image/*");
          imageGetter.launch(pickImageIntent);
        });
  }

  /** Sets up the video mode demo. */
  private void setupVideoModeDemo() {
    imageView = new ObjectDetectionResultImageView(this);
    // The Intent to access gallery and read a video file.
    videoGetter =
        registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
              Intent resultIntent = result.getData();
              if (resultIntent != null) {
                if (result.getResultCode() == RESULT_OK) {
                  MediaMetadataRetriever metaRetriever = new MediaMetadataRetriever();
                  metaRetriever.setDataSource(this, resultIntent.getData());
                  long duration =
                      Long.parseLong(
                          metaRetriever.extractMetadata(
                              MediaMetadataRetriever.METADATA_KEY_DURATION));
                  int numFrames =
                      Integer.parseInt(
                          metaRetriever.extractMetadata(
                              MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));
                  long frameIntervalMs = duration / numFrames;
                  for (int i = 0; i < numFrames; ++i) {
                    MPImage image =
                        new BitmapImageBuilder(metaRetriever.getFrameAtIndex(i)).build();
                    ObjectDetectionResult detectionResult =
                        objectDetector.detectForVideo(image, frameIntervalMs * i);
                    // Currently only annotates the detection result on the first video frame and
                    // display it to verify the correctness.
                    // TODO: Annotates the detection result on every frame, save the
                    // annotated frames as a video file, and play back the video afterwards.
                    if (i == 0) {
                      imageView.setData(image, detectionResult);
                      runOnUiThread(() -> imageView.update());
                    }
                  }
                }
              }
            });
    Button loadVideoButton = findViewById(R.id.button_load_video);
    loadVideoButton.setOnClickListener(
        v -> {
          createObjectDetector(RunningMode.VIDEO);
          updateLayout();
          this.inputSource = InputSource.VIDEO;

          // Reads a video from gallery.
          Intent pickVideoIntent = new Intent(Intent.ACTION_PICK);
          pickVideoIntent.setDataAndType(MediaStore.Video.Media.INTERNAL_CONTENT_URI, "video/*");
          videoGetter.launch(pickVideoIntent);
        });
  }

  private void createObjectDetector(RunningMode mode) {
    if (objectDetector != null) {
      objectDetector.close();
    }
    // Initializes a new MediaPipe ObjectDetector instance
    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_FILE).build())
            .setScoreThreshold(0.5f)
            .setMaxResults(5)
            .setRunningMode(mode)
            .build();
    objectDetector = ObjectDetector.createFromOptions(this, options);
  }

  private void updateLayout() {
    // Updates the preview layout.
    FrameLayout frameLayout = findViewById(R.id.preview_display_layout);
    frameLayout.removeAllViewsInLayout();
    imageView.setImageDrawable(null);
    frameLayout.addView(imageView);
    imageView.setVisibility(View.VISIBLE);
  }

  private Bitmap downscaleBitmap(Bitmap originalBitmap) {
    double aspectRatio = (double) originalBitmap.getWidth() / originalBitmap.getHeight();
    int width = imageView.getWidth();
    int height = imageView.getHeight();
    if (((double) imageView.getWidth() / imageView.getHeight()) > aspectRatio) {
      width = (int) (height * aspectRatio);
    } else {
      height = (int) (width / aspectRatio);
    }
    return Bitmap.createScaledBitmap(originalBitmap, width, height, false);
  }

  private int getImageRotation(InputStream imageData) throws IOException, MediaPipeException {
    int orientation =
        new ExifInterface(imageData)
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
    switch (orientation) {
      case ExifInterface.ORIENTATION_NORMAL:
        return 0;
      case ExifInterface.ORIENTATION_ROTATE_90:
        return 90;
      case ExifInterface.ORIENTATION_ROTATE_180:
        return 180;
      case ExifInterface.ORIENTATION_ROTATE_270:
        return 270;
      default:
        // TODO: use getRotationDegrees() and isFlipped() instead of switch once flip
        // is supported.
        throw new MediaPipeException(
            MediaPipeException.StatusCode.UNIMPLEMENTED.ordinal(),
            "Flipped images are not supported yet.");
    }
  }
}
