"""Starlark rule to create a maven repository from a single artifact."""

_pom_tmpl = """
<?xml version="1.0" encoding="UTF-8"?>
<project xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd"
    xmlns="http://maven.apache.org/POM/4.0.0"',
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
  <modelVersion>4.0.0</modelVersion>
  <groupId>{group_id}</groupId>
  <artifactId>{artifact_id}</artifactId>
  <version>{version}</version>
  <packaging>{packaging}</packaging>
  {identity}
  <licenses>
    <license>
      <name>The Apache Software License, Version 2.0</name>
      <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <developers>
    <developer>
      <name>The MediaPipe Authors</name>
    </developer>
  </developers>
  <dependencies>
    {dependencies}
  </dependencies>
</project>
"""

_identity_tmpl = """
<name>{lib_name}</name>
<description>{lib_description}</description>
<url>{lib_url}</url>
<inceptionYear>{inception_year}</inceptionYear>
"""

_dependency_tmpl = """
<dependency>
  <groupId>{group_id}</groupId>
  <artifactId>{artifact_id}</artifactId>
  <version>{version}</version>
  <scope>compile</scope>
</dependency>
"""

_metadata_tmpl = """
<?xml version="1.0" encoding="UTF-8"?>
<metadata>
  <groupId>{group_id}</groupId>
  <artifactId>{artifact_id}</artifactId>
  <version>{version}</version>
  <versioning>
    <release>{version}</release>
    <versions>
      <version>{version}</version>
    </versions>
    {last_updated_xml}
  </versioning>
</metadata>
"""

def _packaging_type(f):
    """Returns the packaging type used by the file f."""
    if f.basename.endswith(".aar"):
        return "aar"
    elif f.basename.endswith(".jar"):
        return "jar"
    fail("Artifact has unknown packaging type: %s" % f.short_path)

def _create_pom_string(ctx):
    """Returns the contents of the pom file as a string."""
    dependencies = []
    for dep in ctx.attr.artifact_deps:
        if dep.count(":") != 2:
            fail("artifact_deps values must be of form: groupId:artifactId:version")

        group_id, artifact_id, version = dep.split(":")
        dependencies.append(_dependency_tmpl.format(
            group_id = group_id,
            artifact_id = artifact_id,
            version = version,
        ))

    return _pom_tmpl.format(
        group_id = ctx.attr.group_id,
        artifact_id = ctx.attr.artifact_id,
        version = ctx.attr.version,
        packaging = _packaging_type(ctx.file.src),
        identity = _identity_tmpl.format(
            lib_name = ctx.attr.lib_name,
            lib_description = ctx.attr.lib_description,
            lib_url = ctx.attr.lib_url,
            inception_year = ctx.attr.inception_year,
        ),
        dependencies = "\n".join(dependencies),
    )

def _create_metadata_string(ctx):
    """Returns the string contents of maven-metadata.xml for the group."""

    # Include the last_updated string only if provided.
    last_updated_xml = ""
    last_updated = ctx.var.get("MAVEN_ARTIFACT_LAST_UPDATED", "")
    if last_updated != "":
        last_updated_xml = "<lastUpdated>%s</lastUpdated>" % last_updated

    return _metadata_tmpl.format(
        group_id = ctx.attr.group_id,
        artifact_id = ctx.attr.artifact_id,
        version = ctx.attr.version,
        last_updated_xml = last_updated_xml,
    )

def _maven_artifact_impl(ctx):
    """Generates maven repository for a single artifact."""
    pom = ctx.actions.declare_file(
        "%s/%s-%s.pom" % (ctx.label.name, ctx.attr.artifact_id, ctx.attr.version),
    )
    ctx.actions.write(output = pom, content = _create_pom_string(ctx))

    metadata = ctx.actions.declare_file("%s/maven-metadata.xml" % ctx.label.name)
    ctx.actions.write(output = metadata, content = _create_metadata_string(ctx))

    # Rename the artifact to match the naming required inside the repository.
    artifact = ctx.actions.declare_file("%s/%s-%s.%s" % (
        ctx.label.name,
        ctx.attr.artifact_id,
        ctx.attr.version,
        _packaging_type(ctx.file.src),
    ))
    ctx.actions.run_shell(
        inputs = [ctx.file.src],
        outputs = [artifact],
        command = "cp %s %s" % (ctx.file.src.path, artifact.path),
    )

    ctx.actions.run(
        inputs = [pom, metadata, artifact],
        outputs = [ctx.outputs.m2repository],
        arguments = [
            "--group_path=%s" % ctx.attr.group_id.replace(".", "/"),
            "--artifact_id=%s" % ctx.attr.artifact_id,
            "--version=%s" % ctx.attr.version,
            "--artifact=%s" % artifact.path,
            "--pom=%s" % pom.path,
            "--metadata=%s" % metadata.path,
            "--output=%s" % ctx.outputs.m2repository.path,
        ],
        executable = ctx.executable._maven_artifact,
        progress_message = (
            "Packaging repository: %s" % ctx.outputs.m2repository.short_path
        ),
    )

maven_artifact = rule(
    implementation = _maven_artifact_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = [".aar", ".jar"],
        ),
        "group_id": attr.string(mandatory = True),
        "artifact_id": attr.string(mandatory = True),
        "version": attr.string(mandatory = True),
        "last_updated": attr.string(mandatory = True),
        "artifact_deps": attr.string_list(),
        "lib_name": attr.string(default = ""),
        "lib_description": attr.string(default = ""),
        "lib_url": attr.string(default = ""),
        "inception_year": attr.string(default = ""),
        "_maven_artifact": attr.label(
            default = Label("//mediapipe/tasks/internal/release/android:maven_artifact"),
            executable = True,
            allow_files = True,
            cfg = "exec",
        ),
    },
    outputs = {
        "m2repository": "%{name}.zip",
    },
)

def maven_repository(name, srcs):
    """Generates a zip file containing a maven repository."""
    native.genrule(
        name = name,
        srcs = srcs,
        outs = [name + ".zip"],
        cmd = """
        origdir=$$PWD
        TEMP_DIR=$$(mktemp -d)
        for FILE in $(SRCS); do
          unzip $$FILE -d $$TEMP_DIR
        done
        cd $$TEMP_DIR
        zip -r $$origdir/$@ ./
        """,
    )
