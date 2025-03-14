// Copyright (c) 2019, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;
import 'dart:io' show Platform, Directory;

import 'package:path/path.dart' as path;

// FFI signature of the hello_world C function
typedef IncrementFunc = ffi.Int32 Function(ffi.Int32);
typedef Increment = int Function(int);

void main() {
  // Open the dynamic library
  var libraryPath =
      path.join(Directory.current.absolute.path, 'cpp', 'main.dylib');

  // if (Platform.isMacOS) {
  //   libraryPath =
  //       path.join(Directory.current.path, 'hello_library', 'libhello.dylib');
  // }

  // if (Platform.isWindows) {
  //   libraryPath = path.join(
  //       Directory.current.path, 'hello_library', 'Debug', 'hello.dll');
  // }

  final dylib = ffi.DynamicLibrary.open(libraryPath);

  // Look up the C function 'hello_world'
  final Increment increment =
      dylib.lookup<ffi.NativeFunction<IncrementFunc>>('increment').asFunction();
  // Call the function
  print(increment(99));
}
