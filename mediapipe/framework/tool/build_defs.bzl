"""MediaPipe BUILD rules and related utilities."""

# Sanitize a dependency so that it works correctly from targets that
# include MediaPipe as an external dependency.
def clean_dep(dep):
    return str(Label(dep))

# Sanitize a list of dependencies so that they work correctly from targets that
# include MediaPipe as an external dependency.
def clean_deps(dep_list):
    return [clean_dep(dep) for dep in dep_list]
