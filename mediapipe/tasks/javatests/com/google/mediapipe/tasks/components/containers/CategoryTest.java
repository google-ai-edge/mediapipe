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

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.formats.proto.ClassificationProto.Classification;
import com.google.mediapipe.formats.proto.ClassificationProto.ClassificationList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class CategoryTest {

  @Test
  public void create_succeedsWithClassificationProto() {
    Classification input =
        Classification.newBuilder()
            .setScore(0.1f)
            .setIndex(1)
            .setLabel("label")
            .setDisplayName("displayName")
            .build();
    Category output = Category.createFromProto(input);
    assertThat(output.score()).isEqualTo(0.1f);
    assertThat(output.index()).isEqualTo(1);
    assertThat(output.categoryName()).isEqualTo("label");
    assertThat(output.displayName()).isEqualTo("displayName");
  }

  @Test
  public void create_succeedsWithClassificationListProto() {
    Classification element = Classification.newBuilder().setScore(0.1f).build();
    ClassificationList input = ClassificationList.newBuilder().addClassification(element).build();
    List<Category> output = Category.createListFromProto(input);
    assertThat(output).containsExactly(Category.create(0.1f, 0, "", ""));
  }
}
