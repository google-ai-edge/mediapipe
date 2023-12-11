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

package com.google.mediapipe.tasks.components.containers;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.formats.proto.LandmarkProto;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class LandmarkTest {

  @Test
  public void createFromProto_succeedsWithCoordinates() {
    LandmarkProto.Landmark input =
        LandmarkProto.Landmark.newBuilder().setX(1.0f).setY(2.0f).setZ(3.0f).build();
    Landmark output = Landmark.createFromProto(input);
    assertThat(output.x()).isEqualTo(1.0f);
    assertThat(output.y()).isEqualTo(2.0f);
    assertThat(output.z()).isEqualTo(3.0f);
    assertFalse(output.visibility().isPresent());
    assertFalse(output.presence().isPresent());
  }

  @Test
  public void createFromProto_succeedsWithVisibility() {
    LandmarkProto.Landmark input =
        LandmarkProto.Landmark.newBuilder().setVisibility(0.4f).setPresence(0.5f).build();
    Landmark output = Landmark.createFromProto(input);
    assertTrue(output.visibility().isPresent());
    assertThat(output.visibility().get()).isEqualTo(0.4f);
    assertTrue(output.presence().isPresent());
    assertThat(output.presence().get()).isEqualTo(0.5f);
  }

  @Test
  public void createListFromProto_succeeds() {
    LandmarkProto.Landmark element =
        LandmarkProto.Landmark.newBuilder().setX(1.0f).setY(2.0f).setZ(3.0f).build();
    LandmarkProto.LandmarkList input =
        LandmarkProto.LandmarkList.newBuilder().addLandmark(element).build();
    List<Landmark> output = Landmark.createListFromProto(input);
    assertThat(output).hasSize(1);
  }
}
