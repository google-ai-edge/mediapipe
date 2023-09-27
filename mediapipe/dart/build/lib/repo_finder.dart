import 'dart:io' as io;
import 'package:args/args.dart';
import 'package:args/command_runner.dart';
import 'package:path/path.dart' as path;
import 'package:io/ansi.dart';

mixin RepoFinderMixin on Command {
  void addDestinationOption(ArgParser argParser) {
    argParser.addOption(
      'destination',
      abbr: 'd',
      help: 'The location of google/flutter-mediapipe. Defaults to being '
          'adjacent to google/mediapipe.',
    );
  }

  /// Looks upward for the root of the `google/mediapipe` repository. This assumes
  /// the `dart build` command is executed from within said repository. If it is
  /// not executed from within, then this searching algorithm will reach the root
  /// of the file system, log the error, and exit.
  io.Directory findMediaPipeRoot() {
    io.Directory dir = io.Directory(path.current);
    while (true) {
      if (_isMediaPipeRoot(dir)) {
        return dir;
      }
      dir = dir.parent;
      if (dir.parent.path == dir.path) {
        io.stderr.writeln(
          wrapWith(
            'Failed to find google/mediapipe root directory. '
            'Did you execute this command from within the repository?',
            [red],
          ),
        );
        io.exit(1);
      }
    }
  }

  /// Finds the `google/flutter-mediapipe` checkout where artifacts built in this
  /// repository should be copied. By default, this command assumes the two
  /// repositories are siblings on the file system, but the `--destination` flag
  /// allows for this assumption to be overridden.
  io.Directory findFlutterMediaPipeRoot(
    io.Directory mediaPipeDir,
    String? destination,
  ) {
    final flutterMediaPipeDirectory = io.Directory(
      destination ??
          path.joinAll([
            mediaPipeDir.parent.absolute.path,
            'flutter-mediapipe',
          ]),
    );

    if (!flutterMediaPipeDirectory.existsSync()) {
      io.stderr.writeln(
        'Could not find ${flutterMediaPipeDirectory.absolute.path}. '
        'Folder does not exist.',
      );
      io.exit(1);
    }
    return flutterMediaPipeDirectory;
  }
}

/// Looks for 3 files/directories known to be at the root of the google/mediapipe
/// repository. This allows the `dart build` command to be run from various
/// locations within the `google/mediapipe` repository and still correctly set
/// paths for all of its operations.
bool _isMediaPipeRoot(io.Directory dir) {
  final bazelrcExists = io.File(
    path.joinAll(
      [dir.absolute.path, '.bazelrc'],
    ),
  ).existsSync();
  final mediapipeExists = io.Directory(
    path.joinAll(
      [dir.absolute.path, 'mediapipe'],
    ),
  ).existsSync();
  final dotGitExists = io.Directory(
    path.joinAll(
      [dir.absolute.path, '.git'],
    ),
  ).existsSync();
  return bazelrcExists && mediapipeExists && dotGitExists;
}
