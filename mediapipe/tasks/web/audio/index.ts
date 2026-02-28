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

import {AudioClassifier as AudioClassifierImpl} from '../../../tasks/web/audio/audio_classifier/audio_classifier';
import {FilesetResolver as FilesetResolverImpl} from '../../../tasks/web/core/fileset_resolver';

// tslint:disable:enforce-comments-on-exported-symbols

// Declare and export the variables inline so that Rollup in OSS
// explicitly retains the bindings and avoids dead-code elimination bugs.
export const AudioClassifier = AudioClassifierImpl;
export const FilesetResolver = FilesetResolverImpl;
