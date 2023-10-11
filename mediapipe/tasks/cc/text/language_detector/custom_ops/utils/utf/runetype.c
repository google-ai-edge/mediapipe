/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Forked from a library written by Rob Pike and Ken Thompson. Original
// copyright message below.
/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */
#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/utf/utf.h"

static
Rune*
rbsearch(Rune c, Rune *t, int n, int ne)
{
  Rune *p;
  int m;

  while(n > 1) {
    m = n >> 1;
    p = t + m*ne;
    if(c >= p[0]) {
      t = p;
      n = n-m;
    } else
      n = m;
  }
  if(n && c >= t[0])
    return t;
  return 0;
}

#define RUNETYPEBODY
#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/utf/runetypebody.h"
