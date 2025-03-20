"""Build rule to depend on files downloaded from GCS."""

# buildifier: disable=unnamed-macro
def mediapipe_files(srcs):
    """Links file from GCS with the current directory.

    Args:
      srcs: the names of the mediapipe_file target, which is also the name of
      the MediaPipe file in external_files.bzl. For example, if `name` is Foo,
      `mediapipe_file` will create a link to the downloaded file
      "@com_google_mediapipe_Foo_tfile" to the current directory as
      "Foo.tflite".
    """

    for src in srcs:
        archive_name = "com_google_mediapipe_%s" % src.replace("/", "_").replace(".", "_").replace("+", "_")
        native.genrule(
            name = "%s_ln" % archive_name,
            srcs = ["@%s//file" % archive_name],
            outs = [src],
            output_to_bindir = 1,
            cmd = "ln $< $@",
        )
