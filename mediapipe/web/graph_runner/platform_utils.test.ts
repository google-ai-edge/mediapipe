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
import 'jasmine';

import {isWebKit} from '../../web/graph_runner/platform_utils';


const DESKTOP_FIREFOX =
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0';
const DESKTOP_SAFARI =
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Safari/604.1.38,gzip(gfe)';
const IOS_SAFARI =
    'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A334 Safari/7534.48.3';
const IPAD_SAFARI =
    'Mozilla/5.0 (iPad; U; CPU OS 3_2 like Mac OS X; en-us) AppleWebKit/531.21.10 (KHTML, like Gecko) Version/4.0.4 Mobile/7B334b Safari/531.21.10';
const DESKTOP_CHROME =
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/535.8 (KHTML, like Gecko) Chrome/40.0.1000.10 Safari/535.8';
const IOS_CHROME =
    'Mozilla/5.0 (iPhone; CPU iPhone OS 5_1_1 like Mac OS X; en-us) AppleWebKit/534.46.0 (KHTML, like Gecko) CriOS/22.0.1194.0 Mobile/11E53 Safari/7534.48.3';


describe('isWebKit()', () => {
  const navigator = {userAgent: ''};

  it('returns false for Firefox on desktop', () => {
    navigator.userAgent = DESKTOP_FIREFOX;
    expect(isWebKit(navigator as Navigator)).toBeFalse();
  });

  it('returns true for Safari on desktop', () => {
    navigator.userAgent = DESKTOP_SAFARI;
    expect(isWebKit(navigator as Navigator)).toBeTrue();
  });

  it('returns true for Safari on iOS', () => {
    navigator.userAgent = IOS_SAFARI;
    expect(isWebKit(navigator as Navigator)).toBeTrue();
  });

  it('returns true for Safari on iPad', () => {
    navigator.userAgent = IPAD_SAFARI;
    expect(isWebKit(navigator as Navigator)).toBeTrue();
  });

  it('returns false for Chrome on desktop', () => {
    navigator.userAgent = DESKTOP_CHROME;
    expect(isWebKit(navigator as Navigator)).toBeFalse();
  });

  it('returns true for Chrome on iOS', () => {
    navigator.userAgent = IOS_CHROME;
    expect(isWebKit(navigator as Navigator)).toBeTrue();
  });
});
