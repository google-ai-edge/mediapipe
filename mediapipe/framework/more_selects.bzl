"""More utilities to help with selects."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_setting")

# From selects.bzl, but it's not public there.
def _config_setting_always_true(name, visibility):
    """Returns a config_setting with the given name that's always true.

    This is achieved by constructing a two-entry OR chain where each
    config_setting takes opposite values of a bool_setting.
    """
    name_bool = name + "_bool"
    name_on = name + "_bool_on_check"
    name_off = name + "_bool_off_check"
    bool_setting(
        name = name_bool,
        build_setting_default = False,
    )
    native.config_setting(
        name = name_on,
        flag_values = {name_bool: "true"},
    )
    native.config_setting(
        name = name_off,
        flag_values = {name_bool: "false"},
    )
    return selects.config_setting_group(
        name = name,
        visibility = visibility,
        match_any = [
            ":" + name_on,
            ":" + name_off,
        ],
    )

def _config_setting_always_false(name, visibility):
    """Returns a config_setting with the given name that's always false.

    This is achieved by constructing a two-entry AND chain where each
    config_setting takes opposite values of a bool_setting.
    """
    name_bool = name + "_bool"
    name_on = name + "_bool_on_check"
    name_off = name + "_bool_off_check"
    bool_setting(
        name = name_bool,
        build_setting_default = False,
    )
    native.config_setting(
        name = name_on,
        flag_values = {name_bool: "true"},
    )
    native.config_setting(
        name = name_off,
        flag_values = {name_bool: "false"},
    )
    return selects.config_setting_group(
        name = name,
        visibility = visibility,
        match_all = [
            ":" + name_on,
            ":" + name_off,
        ],
    )

def _config_setting_negation(name, negate, visibility = None):
    _config_setting_always_true(
        name = name + "_true",
        visibility = visibility,
    )
    _config_setting_always_false(
        name = name + "_false",
        visibility = visibility,
    )
    native.alias(
        name = name,
        actual = select({
            "//conditions:default": ":%s_true" % name,
            negate: ":%s_false" % name,
        }),
        visibility = visibility,
    )

more_selects = struct(
    config_setting_negation = _config_setting_negation,
)
