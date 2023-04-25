"""MediaPipe Task Library Helper Rules for iOS"""

MPP_TASK_MINIMUM_OS_VERSION = "11.0"

# When the static framework is built with bazel, the all header files are moved
# to the "Headers" directory with no header path prefixes. This auxiliary rule
# is used for stripping the path prefix to the C/iOS API header files included by
# other C/iOS API header files.
# In case of C header files, includes start with a keyword of "#include'.
# Imports in iOS header files start with a keyword of '#import'.
def strip_api_include_path_prefix(name, hdr_labels, prefix = ""):
    """Create modified header files with the import path stripped out.

    Args:
      name: The name to be used as a prefix to the generated genrules.
      hdr_labels: List of header labels to strip out the include path. Each
          label must end with a colon followed by the header file name.
      prefix: Optional prefix path to prepend to the header inclusion path.
    """
    for hdr_label in hdr_labels:
        hdr_filename = hdr_label.split(":")[-1]

        # The last path component of iOS header files is sources/some_file.h
        # Hence it wiill contain a '/'. So the string can be split at '/' to get 
        # the header file name.
        if "/" in hdr_filename:
            hdr_filename = hdr_filename.split("/")[-1]

        hdr_basename = hdr_filename.split(".")[0]
        native.genrule(
            name = "{}_{}".format(name, hdr_basename),
            srcs = [hdr_label],
            outs = [hdr_filename],
            cmd = """
            sed 's|#\\([a-z]*\\) ".*/\\([^/]\\{{1,\\}}\\.h\\)"|#\\1 "{}\\2"|'\
            "$(location {})"\
            > "$@"
            """.format(prefix, hdr_label),
        )

