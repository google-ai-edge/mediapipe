/**
 * Copyright 2022 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** A classification category. */
export declare interface Category {
  /** The probability score of this label category. */
  score: number;

  /** The index of the category in the corresponding label file. */
  index: number;

  /**
   * The label of this category object. Defaults to an empty string if there is
   * no category.
   */
  categoryName: string;

  /**
   * The display name of the label, which may be translated for different
   * locales. For example, a label, "apple", may be translated into Spanish for
   * display purpose, so that the `display_name` is "manzana". Defaults to an
   * empty string if there is no display name.
   */
  displayName: string;
}
