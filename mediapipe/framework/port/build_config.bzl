# TODO: Split build rules into several individual files to make them
# more manageable.

""".bzl file for mediapipe open source build configs."""

load(
    "//mediapipe/framework/tool:mediapipe_proto.bzl",
    _mediapipe_cc_proto_library = "mediapipe_cc_proto_library",
    _mediapipe_proto_library = "mediapipe_proto_library",
)

# TODO: enable def_rewrite once alias proto sources are generated.
def mediapipe_proto_library(def_rewrite = False, **kwargs):
    _mediapipe_proto_library(def_rewrite = def_rewrite, **kwargs)

mediapipe_cc_proto_library = _mediapipe_cc_proto_library

def provided_args(**kwargs):
    """Returns the keyword arguments omitting None arguments."""
    return {k: v for k, v in kwargs.items() if v != None}

def replace_suffix(string, old, new):
    """Returns a string with an old suffix replaced by a new suffix."""
    return string.endswith(old) and string[:-len(old)] + new or string

