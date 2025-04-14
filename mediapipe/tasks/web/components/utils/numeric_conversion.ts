/**
 * Copyright 2025 The MediaPipe Authors.
 *
 * <p>Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * <p>http://www.apache.org/licenses/LICENSE-2.0
 *
 * <p>Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

// Historically, JSPB getters for 64-bit int fields would return a `number`
// typed value when within the safe range and a string otherwise to retain
// precision.
export function asLegacyNumberOrString(value: unknown): number {
  const num = Number(value);
  return Number.isSafeInteger(num) ? num : (String(value) as unknown as number);
}
