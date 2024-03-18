"""Rules implementation for mediapipe_proto_alias.bzl, do not load directly."""

def _copy_header_impl(ctx):
    source = ctx.attr.source.replace("//", "").replace(":", "/")
    files = []
    for dep in ctx.attr.deps:
        for header in dep[CcInfo].compilation_context.direct_headers:
            if (header.short_path == source):
                files.append(header)
    if len(files) != 1:
        fail("Expected exactly 1 source, got ", str(files))
    dest_file = ctx.actions.declare_file(ctx.attr.filename)

    # Use expand_template() with no substitutions as a simple copier.
    ctx.actions.expand_template(
        template = files[0],
        output = dest_file,
        substitutions = {},
    )
    return [DefaultInfo(files = depset([dest_file]))]

copy_header = rule(
    implementation = _copy_header_impl,
    attrs = {
        "filename": attr.string(),
        "source": attr.string(),
        "deps": attr.label_list(providers = [CcInfo]),
    },
    outputs = {"out": "%{filename}"},
)
