import 'dart:async';
import 'dart:io' as io;
import 'package:io/ansi.dart' as ansi;
import 'package:path/path.dart' as path;
import 'commands.dart';

class DartProtoBuilder {
  DartProtoBuilder([this.options = const DartProtoBuilderOptions()])
      : repositoryRoot = io.Directory.current.parent.parent.parent {
    // Running this command from within bin (aka: `dart dart_builder.dart`)
    // throws everything off by one level. This merges that invocation style
    // with the expected, which is `dart bin/dart_builder.dart`.
    repositoryRoot = repositoryRoot.absolute.path
            .endsWith('mediapipe${io.Platform.pathSeparator}mediapipe')
        ? repositoryRoot.parent
        : repositoryRoot;

    // Assert we correctly calculated the repository root.
    if (!io.Directory(
      path.join(repositoryRoot.absolute.path, '.git'),
    ).existsSync()) {
      io.stderr.writeAll(
        [
          'Executed dart_builder.dart from unexpected directory.\n',
          'Try running: `dart bin${io.Platform.pathSeparator}dart_builder.dart`\n'
        ],
      );
      io.exit(1);
    }

    _mediapipeDir = io.Directory(
      path.join(repositoryRoot.absolute.path, 'mediapipe'),
    );

    _dartBuilderDirectory = io.Directory(
      path.join(
          repositoryRoot.absolute.path, 'mediapipe', 'dart', 'dart_builder'),
    );

    _outputDirectory = options.outputPath != null
        ? io.Directory(options.outputPath!)
        : io.Directory(
            path.join(
              repositoryRoot.absolute.path,
              'mediapipe',
              'dart',
              'mediapipe',
              'lib',
              'generated',
            ),
          );
  }

  final DartProtoBuilderOptions options;

  io.Directory repositoryRoot;

  // This is the `mediapipe` directory *within* the repository itself, which
  // is also named `mediapipe`
  io.Directory get mediapipeDir => _mediapipeDir!;
  io.Directory? _mediapipeDir;

  /// Directory to place compiled protobufs.
  io.Directory get outputDirectory => _outputDirectory!;
  io.Directory? _outputDirectory;

  /// Location of this command.
  io.Directory get dartBuilderDirectory => _dartBuilderDirectory!;
  io.Directory? _dartBuilderDirectory;

  /// Location of local copy of git@github.com:protocolbuffers/protobuf.git.
  ///
  /// This can either be passed in via `options`, or the script will check for
  /// the repository as a sibling to where `mediapipe` is checked out.
  ///
  /// This starts out as `null` but is set by `_confirmProtobufDirectory`, which
  /// either successfully locates the repository or exits.
  io.Directory get protobufDirectory => _protobufDirectory!;
  io.Directory? _protobufDirectory;

  final _protosToCompile = <io.File>[];

  Future<void> run() async {
    await _confirmOutputDirectories();
    await _confirmProtocPlugin();
    await _confirmProtoc();
    await _confirmProtocGenDart();
    await _confirmProtobufDirectory();
    io.stdout.writeln(
      ansi.wrapWith(
        'Dependencies installed correctly.',
        [ansi.green],
      ),
    );
    await _buildProtos();
    await _buildBarrelFiles();
  }

  Future<void> _buildProtos() async {
    await _prepareProtos();

    await for (io.FileSystemEntity entity
        in mediapipeDir.list(recursive: true)) {
      if (entity is! io.File) continue;
      if (entity.path.endsWith('test.proto')) continue;
      if (entity.path.contains('tensorflow')) continue;
      if (entity.path.contains('testdata')) continue;
      if (entity.path.contains('examples')) continue;
      if (!entity.path.endsWith('.proto')) continue;
      _protosToCompile.add(entity);
    }

    // Lastly, MediaPipe's protobufs have 1 dependency on the `Any` protobuf
    // from `google/protobuf`. Thus, for the whole thing to compile, we need to
    // add just that class.
    final outsideRepository = io.Directory(repositoryRoot.parent.absolute.path);
    final anyProto = io.File(
      path.join(
        outsideRepository.absolute.path,
        'protobuf',
        'src',
        'google',
        'protobuf',
        'any.proto',
      ),
    );
    _protosToCompile.add(anyProto);
    _compileProtos(_protosToCompile);
  }

  Future<void> _buildBarrelFiles() async {
    final generatedOutput = _GeneratedOutput();
    final fileSystemWalk = outputDirectory.list(recursive: true);

    await for (io.FileSystemEntity entity in fileSystemWalk) {
      if (entity is io.Directory) {
        generatedOutput.add(entity);
      } else if (entity is io.File) {
        generatedOutput.addFile(entity.parent, entity);
      }
    }

    int numBarrelFilesGenerated = 0;
    final sep = io.Platform.pathSeparator;
    for (final gen in generatedOutput.getDirectories()) {
      final fileExports = StringBuffer();
      final nestedBarrelExports = StringBuffer();

      for (final dir in gen.directories) {
        final dirName = dir.absolute.path.split(sep).last;
        nestedBarrelExports.writeln("export '$dirName/$dirName.dart';");
      }

      for (final file in gen.files) {
        final fileName = file.absolute.path.split(sep).last;

        if (!fileName.endsWith('.dart')) continue;

        // The top-level mediapipe.dart barrel file surfaces multiple collisions
        // from classes that exist in various libraries and then again within
        // `tasks`. To avoid these collisions, we skip generating the top-level
        // barrel file.
        if (fileName == 'mediapipe.dart') continue;

        // Skip barrel files generated from a previous run
        final parentFolderName = file.parent.absolute.path.split(sep).last;
        if (fileName.split('.').first == parentFolderName) continue;

        // Finally, add the valid file export
        fileExports.writeln("export '$fileName';");
      }

      final hostDirName = gen.dir.absolute.path.split(sep).last;
      final barrelFile = io.File(
        '${gen.dir.absolute.path}$sep$hostDirName.dart',
      );
      // Delete a pre-existing barrel file.
      if (barrelFile.existsSync()) {
        io.File(barrelFile.absolute.path).deleteSync();
      }
      // Write to our new file.
      final spacingNewline = nestedBarrelExports.isNotEmpty ? '\n' : '';
      barrelFile.writeAsStringSync(
        '${nestedBarrelExports.toString()}$spacingNewline${fileExports.toString()}\n',
      );
      numBarrelFilesGenerated++;
    }
    io.stdout.writeln(
      ansi.wrapWith(
        'Generated $numBarrelFilesGenerated barrel files successfully.',
        [ansi.green],
      ),
    );
  }

  String get protocPath => options.protocPath ?? 'protoc';

  /// Builds a single protobuf definition's Dart file.
  Future<void> _compileProtos(List<io.File> protoFiles) async {
    final command = Command.run(
      [
        protocPath,
        '-I${repositoryRoot.absolute.path}',
        '-I$_protobufCompilationImportPath',
        '--dart_out=grpc:${outputDirectory.absolute.path}',
        ...protoFiles.map<String>((entity) => entity.absolute.path),
      ],
      workingDirectory: dartBuilderDirectory.absolute.path,
      debug: true,
    );
    await command.run();
    if (command.exitCode! != 0) {
      io.stderr.writeln('Exiting with ${command.exitCode}');
      io.exit(command.exitCode!);
    }
  }

  String get _protobufCompilationImportPath => io.Directory(
        path.join(protobufDirectory.absolute.path, 'src'),
      ).absolute.path;

  Future<void> _confirmOutputDirectories() async {
    if (!await _outputDirectory!.exists()) {
      _outputDirectory!.create();
    }
  }

  Future<void> _prepareProtos() async {
    if (!await outputDirectory.exists()) {
      io.stdout.writeln('Creating output directory');
      outputDirectory.create();
    }
  }

  Future<void> _confirmProtoc() async => //
      Command.which(
        protocPath,
        documentationUrl:
            'http://google.github.io/proto-lens/installing-protoc.html',
      ).run();

  Future<void> _confirmProtocGenDart() async => //
      Command.which(
        'protoc-gen-dart',
        documentationUrl:
            'http://google.github.io/proto-lens/installing-protoc.html',
      ).run();

  Future<void> _confirmProtobufDirectory() async {
    _protobufDirectory = options.protobufPath != null
        ? io.Directory(options.protobufPath!)
        : io.Directory(
            path.join(repositoryRoot.parent.absolute.path, 'protobuf'),
          );
    if (!await _protobufDirectory!.exists()) {
      io.stderr.writeAll(
        [
          'Could not find google/protobuf repository. You can clone this from ',
          'https://github.com/protocolbuffers/protobuf and either pass its '
              'location via the `--protobuf` flag, or by default, clone it in the '
              'same directory where you cloned `google/mediapipe`.\n'
              '\n',
          'Checked for protobuf library at ${_protobufDirectory!.absolute.path}\n',
        ],
      );
      io.exit(1);
    }
  }

  Future<void> _confirmProtocPlugin() async =>
      Command.needleInOutput(['dart', 'pub', 'global', 'list'],
          needle: 'protoc_plugin',
          ifMissing: [
            'protoc_plugin does not seem to be installed',
            'Run `dart pub global activate protoc_plugin` to install it',
          ]).run();
}

class DartProtoBuilderOptions {
  const DartProtoBuilderOptions({
    this.protocPath,
    this.protobufPath,
    this.outputPath,
  });

  /// Location of `protoc`. Defaults to whatever is found on $PATH.
  final String? protocPath;

  /// Location of local copy of git@github.com:protocolbuffers/protobuf.git
  final String? protobufPath;

  /// Place to put the generated protobufs.
  final String? outputPath;
}

/// Bundles a set of _GeneratedOutputDirectory objects with helpful getters.
class _GeneratedOutput {
  final Map<String, _GeneratedOutputDirectory> generated = {};

  void add(io.Directory dir) {
    generated[dir.absolute.path] = _GeneratedOutputDirectory(dir);
    if (contains(dir.parent)) {
      generated[dir.parent.absolute.path]!.addChild(dir);
    }
  }

  bool contains(io.Directory dir) => generated.containsKey(dir.absolute.path);

  _GeneratedOutputDirectory get(io.Directory dir) {
    if (!contains(dir)) {
      throw Exception('Unexpectedly asked for unknown directory: '
          '${dir.absolute.path}. Are you recursively walking the file system?');
    }
    return generated[dir.absolute.path]!;
  }

  void addFile(io.Directory dir, io.File file) => get(dir).addFile(file);

  Iterable<_GeneratedOutputDirectory> getDirectories() sync* {
    for (final gen in generated.values) {
      yield gen;
    }
  }
}

/// Tracks which Dart files were generated where for the purposes of adding
/// barrel files.
class _GeneratedOutputDirectory {
  _GeneratedOutputDirectory(this.dir);
  final io.Directory dir;
  final List<io.Directory> directories = [];
  final List<io.File> files = [];

  void addChild(io.Directory dir) => directories.add(dir);
  void addFile(io.File file) => files.add(file);
}
