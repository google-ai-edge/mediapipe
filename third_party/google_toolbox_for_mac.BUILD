# Description:
# A collection of source from different Google projects that may be of use to
# developers working other Mac projects.

# To run all the test from the command line:
#   bazel test \
#       --build_tests_only \
#       @google_toolbox_for_mac///...

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

exports_files(
    ["UnitTest-Info.plist"],
    visibility = ["//visibility:public"],
)

objc_library(
    name = "GTM_Defines",
    hdrs = ["GTMDefines.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

objc_library(
    name = "GTM_TypeCasting",
    hdrs = [
        "DebugUtils/GTMTypeCasting.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_LocalizedString",
    hdrs = [
        "Foundation/GTMLocalizedString.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_NSStringURLArguments",
    srcs = [
        "Foundation/GTMNSString+URLArguments.m",
    ],
    hdrs = [
        "Foundation/GTMNSString+URLArguments.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_GeometryUtils",
    srcs = [
        "Foundation/GTMGeometryUtils.m",
    ],
    hdrs = [
        "Foundation/GTMGeometryUtils.h",
    ],
    sdk_frameworks = ["CoreGraphics"],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

# Since this is just .h files, it is ok to not divide this into sub targets as it
# doesn't cause any extra code to be linked in when some just wants a subset of
# it.
objc_library(
    name = "GTM_DebugUtils",
    hdrs = [
        "DebugUtils/GTMDebugSelectorValidation.h",
        "DebugUtils/GTMDebugThreadValidation.h",
        "DebugUtils/GTMMethodCheck.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_SynchronizationAsserts",
    srcs = [
        "DebugUtils/GTMSynchronizationAsserts.m",
    ],
    hdrs = [
        "DebugUtils/GTMSynchronizationAsserts.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_KVO",
    hdrs = [
        "Foundation/GTMNSObject+KeyValueObserving.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMNSObject+KeyValueObserving.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_DebugUtils",
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_Regex",
    hdrs = [
        "Foundation/GTMRegex.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMRegex.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_StringEncoding",
    hdrs = [
        "Foundation/GTMStringEncoding.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMStringEncoding.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_NSScannerJSON",
    srcs = [
        "Foundation/GTMNSScanner+JSON.m",
    ],
    hdrs = [
        "Foundation/GTMNSScanner+JSON.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_NSStringHTML",
    hdrs = [
        "Foundation/GTMNSString+HTML.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMNSString+HTML.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_NSStringXML",
    srcs = [
        "Foundation/GTMNSString+XML.m",
    ],
    hdrs = [
        "Foundation/GTMNSString+XML.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_NSThreadBlocks",
    hdrs = [
        "Foundation/GTMNSThread+Blocks.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMNSThread+Blocks.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_TimeUtils",
    hdrs = [
        "Foundation/GTMTimeUtils.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMTimeUtils.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_SQLite",
    hdrs = [
        "Foundation/GTMSQLite.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMSQLite.m",
    ],
    sdk_dylibs = ["libsqlite3"],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_DebugUtils",
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_SystemVersion",
    hdrs = [
        "Foundation/GTMSystemVersion.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMSystemVersion.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_GTMURLBuilder",
    hdrs = [
        "Foundation/GTMURLBuilder.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMURLBuilder.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Logger",
        ":GTM_NSDictionaryURLArguments",
    ],
)

objc_library(
    name = "GTM_NSDictionaryURLArguments",
    srcs = [
        "Foundation/GTMNSDictionary+URLArguments.m",
    ],
    hdrs = [
        "Foundation/GTMNSDictionary+URLArguments.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_DebugUtils",
        ":GTM_NSStringURLArguments",
    ],
)

objc_library(
    name = "GTM_StackTrace",
    hdrs = [
        "Foundation/GTMStackTrace.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMStackTrace.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_NSDataZlib",
    srcs = [
        "Foundation/GTMNSData+zlib.m",
    ],
    hdrs = [
        "Foundation/GTMNSData+zlib.h",
    ],
    sdk_dylibs = [
        "libz",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_NSFileHandleUniqueName",
    hdrs = [
        "Foundation/GTMNSFileHandle+UniqueName.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMNSFileHandle+UniqueName.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_UIFontLineHeight",
    srcs = [
        "iPhone/GTMUIFont+LineHeight.m",
    ],
    hdrs = [
        "iPhone/GTMUIFont+LineHeight.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_RoundedRectPath",
    srcs = [
        "iPhone/GTMRoundedRectPath.m",
    ],
    hdrs = [
        "iPhone/GTMRoundedRectPath.h",
    ],
    sdk_frameworks = ["CoreGraphics"],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_UIImageResize",
    srcs = [
        "iPhone/GTMUIImage+Resize.m",
    ],
    hdrs = [
        "iPhone/GTMUIImage+Resize.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_FadeTruncatingLabel",
    hdrs = [
        "iPhone/GTMFadeTruncatingLabel.h",
    ],
    non_arc_srcs = [
        "iPhone/GTMFadeTruncatingLabel.m",
    ],
    visibility = ["//visibility:public"],
)

objc_library(
    name = "GTM_UILocalizer",
    hdrs = select({
        "//mediapipe:macos": ["AppKit/GTMUILocalizer.h"],
        "//conditions:default": ["iPhone/GTMUILocalizer.h"],
    }),
    non_arc_srcs = select({
        "//mediapipe:macos": ["AppKit/GTMUILocalizer.m"],
        "//conditions:default": ["iPhone/GTMUILocalizer.m"],
    }),
    sdk_frameworks = select({
        "//mediapipe:macos": ["AppKit"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
    # On MacOS, mark alwayslink in case this is referenced only from a XIB and
    # would otherwise be stripped.
    alwayslink = select({
        "//mediapipe:ios": 0,
        "//conditions:default": 1,
    }),
)

# NOTE: This target is only available for MacOS, not iPhone.
objc_library(
    name = "GTM_UILocalizerAndLayoutTweaker",
    hdrs = ["AppKit/GTMUILocalizerAndLayoutTweaker.h"],
    non_arc_srcs = ["AppKit/GTMUILocalizerAndLayoutTweaker.m"],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
        ":GTM_UILocalizer",
    ],
    # Mark alwayslink in case this is referenced only from a XIB and would
    # otherwise be stripped.
    alwayslink = 1,
)

GTM_UNITTESTING_HDRS = [
    "UnitTesting/GTMFoundationUnitTestingUtilities.h",
    "UnitTesting/GTMSenTestCase.h",
    "UnitTesting/GTMTestTimer.h",
]

GTM_UNITTESTING_NON_ARC_SRCS = [
    "UnitTesting/GTMFoundationUnitTestingUtilities.m",
    "UnitTesting/GTMSenTestCase.m",
]

GTM_UNITTESTING_SDK_FRAMEWORKS = [
    "CoreGraphics",
    "QuartzCore",
]

GTM_UNITTESTING_DEPS = [
    ":GTM_Regex",
    ":GTM_SystemVersion",
]

objc_library(
    name = "GTM_UnitTesting",
    testonly = True,
    hdrs = GTM_UNITTESTING_HDRS,
    non_arc_srcs = GTM_UNITTESTING_NON_ARC_SRCS,
    sdk_frameworks = GTM_UNITTESTING_SDK_FRAMEWORKS,
    visibility = ["//visibility:public"],
    deps = GTM_UNITTESTING_DEPS,
)

objc_library(
    name = "GTM_UnitTesting_GTM_USING_XCTEST",
    testonly = True,
    hdrs = GTM_UNITTESTING_HDRS,
    defines = ["GTM_USING_XCTEST=1"],
    non_arc_srcs = GTM_UNITTESTING_NON_ARC_SRCS,
    sdk_frameworks = GTM_UNITTESTING_SDK_FRAMEWORKS,
    visibility = ["//visibility:public"],
    deps = GTM_UNITTESTING_DEPS,
)

objc_library(
    name = "GTM_UnitTestingAppLib",
    testonly = True,
    hdrs = [
        "UnitTesting/GTMCodeCoverageApp.h",
        "UnitTesting/GTMIPhoneUnitTestDelegate.h",
    ],
    non_arc_srcs = [
        "UnitTesting/GTMIPhoneUnitTestDelegate.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_UnitTesting",
    ],
)

# No Test for GTM_UnitTestingAppLib, use a build test.
objc_library(
    name = "GTM_Logger",
    hdrs = [
        "Foundation/GTMLogger.h",
    ],
    non_arc_srcs = [
        "Foundation/GTMLogger.m",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":GTM_Defines",
    ],
)

objc_library(
    name = "GTM_Logger_ASL",
    hdrs = ["Foundation/GTMLogger+ASL.h"],
    non_arc_srcs = ["Foundation/GTMLogger+ASL.m"],
    visibility = ["//visibility:public"],
    deps = [":GTM_Logger"],
)

objc_library(
    name = "GTM_LoggerRingBufferWriter",
    hdrs = ["Foundation/GTMLoggerRingBufferWriter.h"],
    non_arc_srcs = ["Foundation/GTMLoggerRingBufferWriter.m"],
    visibility = ["//visibility:public"],
    deps = [":GTM_Logger"],
)
