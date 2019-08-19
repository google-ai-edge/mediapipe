# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

exports_files(["LICENSE"])

config_setting(
    name = "android_arm",
    values = {
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "android_arm64",
    values = {
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "ios_armv7",
    values = {
        "cpu": "ios_armv7",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "ios_arm64",
    values = {
        "cpu": "ios_arm64",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "ios_arm64e",
    values = {
        "cpu": "ios_arm64e",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "libunwind",
    values = {
        "define": "libunwind=true",
        "cpu": "k8",
    },
    visibility = ["//visibility:private"],
)

cc_library(
    name = "glog",
    srcs = [
        "config_h",
        "src/base/commandlineflags.h",
        "src/base/googleinit.h",
        "src/base/mutex.h",
        "src/demangle.cc",
        "src/demangle.h",
        "src/logging.cc",
        "src/raw_logging.cc",
        "src/signalhandler.cc",
        "src/symbolize.cc",
        "src/symbolize.h",
        "src/utilities.cc",
        "src/utilities.h",
        "src/vlog_is_on.cc",
    ] + glob(["src/stacktrace*.h"]),
    hdrs = [
        "src/glog/log_severity.h",
        "src/glog/logging.h",
        "src/glog/raw_logging.h",
        "src/glog/stl_logging.h",
        "src/glog/vlog_is_on.h",
    ],
    copts = [
        "-Wno-sign-compare",
        "-U_XOPEN_SOURCE",
    ],
    includes = ["./src"],
    linkopts = select({
        ":libunwind": ["-lunwind"],
        "//conditions:default": [],
    }) + select({
        "//conditions:default": ["-lpthread"],
        ":android_arm": [],
        ":android_arm64": [],
        ":ios_armv7": [],
        ":ios_arm64": [],
        ":ios_arm64e": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": ["@com_github_gflags_gflags//:gflags"],
        ":android_arm": [],
        ":android_arm64": [],
        ":ios_armv7": [],
        ":ios_arm64": [],
        ":ios_arm64e": [],
    }),
)

genrule(
    name = "run_configure",
    srcs = [
        "README",
        "Makefile.in",
        "config.guess",
        "config.sub",
        "install-sh",
        "ltmain.sh",
        "missing",
        "libglog.pc.in",
        "src/config.h.in",
        "src/glog/logging.h.in",
        "src/glog/raw_logging.h.in",
        "src/glog/stl_logging.h.in",
        "src/glog/vlog_is_on.h.in",
    ],
    outs = [
        "config.h.tmp",
        "src/glog/logging.h.tmp",
        "src/glog/raw_logging.h",
        "src/glog/stl_logging.h",
        "src/glog/vlog_is_on.h",
    ],
    cmd = "$(location :configure)" +
          "&& cp -v src/config.h $(location config.h.tmp) " +
          "&& cp -v src/glog/logging.h $(location src/glog/logging.h.tmp) " +
          "&& cp -v src/glog/raw_logging.h $(location src/glog/raw_logging.h) " +
          "&& cp -v src/glog/stl_logging.h $(location src/glog/stl_logging.h) " +
          "&& cp -v src/glog/vlog_is_on.h $(location src/glog/vlog_is_on.h) ",
    tools = [
        "configure",
    ],
)

genrule(
    name = "config_h",
    srcs = select({
        "//conditions:default": ["config.h.tmp"],
        ":android_arm": ["config.h.android_arm"],
        ":android_arm64": ["config.h.android_arm"],
        ":ios_armv7": ["config.h.ios_arm"],
        ":ios_arm64": ["config.h.ios_arm"],
        ":ios_arm64e": ["config.h.ios_arm"],
    }),
    outs = ["config.h"],
    cmd = "echo select $< to be the glog config file. && cp $< $@",
)

genrule(
    name = "logging_h",
    srcs = select({
        "//conditions:default": ["src/glog/logging.h.tmp"],
        ":android_arm": ["src/glog/logging.h.arm"],
        ":android_arm64": ["src/glog/logging.h.arm"],
        ":ios_armv7": ["src/glog/logging.h.arm"],
        ":ios_arm64": ["src/glog/logging.h.arm"],
        ":ios_arm64e": ["src/glog/logging.h.arm"],
    }),
    outs = ["src/glog/logging.h"],
    cmd = "echo select $< to be the glog logging.h file. && cp $< $@",
)

# Hardcoded android arm config header for glog library.
# TODO: This is a temporary workaround. We should generate the config
# header by running the configure script with the right target toolchain.
ANDROID_ARM_CONFIG = """
/* Define if glog does not use RTTI */
/* #undef DISABLE_RTTI */

/* Namespace for Google classes */
#define GOOGLE_NAMESPACE google

/* Define if you have the 'dladdr' function */
#define HAVE_DLADDR

/* Define if you have the 'snprintf' function */
#define HAVE_SNPRINTF

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H

/* Define to 1 if you have the <execinfo.h> header file. */
/* #undef HAVE_EXECINFO_H */

/* Define if you have the 'fcntl' function */
#define HAVE_FCNTL

/* Define to 1 if you have the <glob.h> header file. */
#define HAVE_GLOB_H

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the 'pthread' library (-lpthread). */
/* #undef HAVE_LIBPTHREAD */

/* Define to 1 if you have the <libunwind.h> header file. */
/* #undef HAVE_LIBUNWIND_H */

/* Define if you have google gflags library */
/* #undef HAVE_LIB_GFLAGS */

/* Define if you have google gmock library */
/* #undef HAVE_LIB_GMOCK */

/* Define if you have google gtest library */
/* #undef HAVE_LIB_GTEST */

/* Define if you have libunwind */
/* #undef HAVE_LIB_UNWIND */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H

/* Define to disable multithreading support. */
/* #undef NO_THREADS */

/* Define if the compiler implements namespaces */
#define HAVE_NAMESPACES

/* Define if you have the 'pread' function */
#define HAVE_PREAD

/* Define if you have POSIX threads libraries and header files. */
#define HAVE_PTHREAD

/* Define to 1 if you have the <pwd.h> header file. */
#define HAVE_PWD_H

/* Define if you have the 'pwrite' function */
#define HAVE_PWRITE

/* Define if the compiler implements pthread_rwlock_* */
#define HAVE_RWLOCK

/* Define if you have the 'sigaction' function */
#define HAVE_SIGACTION

/* Define if you have the 'sigaltstack' function */
/* #undef HAVE_SIGALTSTACK */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H

/* Define to 1 if you have the <syscall.h> header file. */
#define HAVE_SYSCALL_H

/* Define to 1 if you have the <syslog.h> header file. */
#define HAVE_SYSLOG_H

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H

/* Define to 1 if you have the <sys/syscall.h> header file. */
#define HAVE_SYS_SYSCALL_H

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/ucontext.h> header file. */
/* #undef HAVE_SYS_UCONTEXT_H */

/* Define to 1 if you have the <sys/utsname.h> header file. */
#define HAVE_SYS_UTSNAME_H

/* Define to 1 if you have the <ucontext.h> header file. */
#define HAVE_UCONTEXT_H

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the <unwind.h> header file. */
#define HAVE_UNWIND_H 1

/* Define if the compiler supports using expression for operator */
#define HAVE_USING_OPERATOR

/* Define if your compiler has __attribute__ */
#define HAVE___ATTRIBUTE__

/* Define if your compiler has __builtin_expect */
#define HAVE___BUILTIN_EXPECT 1

/* Define if your compiler has __sync_val_compare_and_swap */
#define HAVE___SYNC_VAL_COMPARE_AND_SWAP

/* Define to the sub-directory in which libtool stores uninstalled libraries. */

/* #undef LT_OBJDIR */

/* Name of package */
/* #undef PACKAGE */

/* Define to the address where bug reports for this package should be sent. */
/* #undef PACKAGE_BUGREPORT */

/* Define to the full name of this package. */
/* #undef PACKAGE_NAME */

/* Define to the full name and version of this package. */
/* #undef PACKAGE_STRING */

/* Define to the one symbol short name of this package. */
/* #undef PACKAGE_TARNAME */

/* Define to the home page for this package. */
/* #undef PACKAGE_URL */

/* Define to the version of this package. */
/* #undef PACKAGE_VERSION */

/* How to access the PC from a struct ucontext */
/* #undef PC_FROM_UCONTEXT */

/* Define to necessary symbol if this constant uses a non-standard name on
your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* The size of , as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* Define to 1 if you have the ANSI C header files. */
/* #undef STDC_HEADERS */

/* the namespace where STL code like vector<> is defined */
#define STL_NAMESPACE std

/* Version number of package */
/* #undef VERSION */

/* Stops putting the code inside the Google namespace */
#define _END_GOOGLE_NAMESPACE_ }

/* Puts following code inside the Google namespace */
#define _START_GOOGLE_NAMESPACE_ namespace google {
"""

genrule(
    name = "gen_android_arm_config",
    outs = ["config.h.android_arm"],
    cmd = ("echo '%s' > $(location config.h.android_arm)" % ANDROID_ARM_CONFIG),
)

genrule(
    name = "generate_arm_glog_logging_h",
    srcs = ["src/glog/logging.h.in"],
    outs = ["src/glog/logging.h.arm"],
    cmd = ("sed -e 's/@ac_cv___attribute___noinline@/__attribute__((__noinline__))/g'" +
           "    -e 's/@ac_cv___attribute___noreturn@/__attribute__((__noreturn__))/g'" +
           "    -e 's/@ac_cv_have___builtin_expect@/1/g'" +
           "    -e 's/@ac_cv_have___uint16@/0/g'" +
           "    -e 's/@ac_cv_have_inttypes_h@/1/g'" +
           "    -e 's/@ac_cv_have_libgflags@/0/g'" +
           "    -e 's/@ac_cv_have_stdint_h@/1/g'" +
           "    -e 's/@ac_cv_have_systypes_h@/1/g'" +
           "    -e 's/@ac_cv_have_u_int16_t@/0/g'" +
           "    -e 's/@ac_cv_have_uint16_t@/1/g'" +
           "    -e 's/@ac_cv_have_unistd_h@/1/g'" +
           "    -e 's/@ac_google_end_namespace@/}/g'" +
           "    -e 's/@ac_google_namespace@/google/g'" +
           "    -e 's/@ac_google_start_namespace@/namespace google {/g'" +
           "  $< > $@"),
)

# Hardcoded ios arm config header for glog library.
# TODO: This is a temporary workaround. We should generate the config
# header by running the configure script with the right target toolchain.
IOS_ARM_CONFIG = """
/* define if glog doesnt use RTTI */
/* #undef DISABLE_RTTI */

/* Namespace for Google classes */
#define GOOGLE_NAMESPACE google

/* Define if you have the 'dladdr' function */
#define HAVE_DLADDR 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <execinfo.h> header file. */
#define HAVE_EXECINFO_H 1

/* Define if you have the 'fcntl' function */
#define HAVE_FCNTL 1

/* Define to 1 if you have the <glob.h> header file. */
#define HAVE_GLOB_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the 'pthread' library (-lpthread). */
#define HAVE_LIBPTHREAD 1

/* Define to 1 if you have the <libunwind.h> header file. */
#define HAVE_LIBUNWIND_H 1

/* define if you have google gflags library */
/* #undef HAVE_LIB_GFLAGS */

/* define if you have google gmock library */
/* #undef HAVE_LIB_GMOCK */

/* define if you have google gtest library */
/* #undef HAVE_LIB_GTEST */

/* define if you have libunwind */
/* #undef HAVE_LIB_UNWIND */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* define if the compiler implements namespaces */
#define HAVE_NAMESPACES 1

/* Define if you have the 'pread' function */
#define HAVE_PREAD 1

/* Define if you have POSIX threads libraries and header files. */
#define HAVE_PTHREAD 1

/* Define to 1 if you have the <pwd.h> header file. */
#define HAVE_PWD_H 1

/* Define if you have the 'pwrite' function */
#define HAVE_PWRITE 1

/* define if the compiler implements pthread_rwlock_* */
#define HAVE_RWLOCK 1

/* Define if you have the 'sigaction' function */
#define HAVE_SIGACTION 1

/* Define if you have the 'sigaltstack' function */
#define HAVE_SIGALTSTACK 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <syscall.h> header file. */
/* #undef HAVE_SYSCALL_H */

/* Define to 1 if you have the <syslog.h> header file. */
#define HAVE_SYSLOG_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/syscall.h> header file. */
#define HAVE_SYS_SYSCALL_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/ucontext.h> header file. */
#define HAVE_SYS_UCONTEXT_H 1

/* Define to 1 if you have the <sys/utsname.h> header file. */
#define HAVE_SYS_UTSNAME_H 1

/* Define to 1 if you have the <ucontext.h> header file. */
/* #undef HAVE_UCONTEXT_H */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the <unwind.h> header file. */
#define HAVE_UNWIND_H 1

/* define if the compiler supports using expression for operator */
#define HAVE_USING_OPERATOR 1

/* define if your compiler has __attribute__ */
#define HAVE___ATTRIBUTE__ 1

/* define if your compiler has __builtin_expect */
#define HAVE___BUILTIN_EXPECT 1

/* define if your compiler has __sync_val_compare_and_swap */
#define HAVE___SYNC_VAL_COMPARE_AND_SWAP 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "glog"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "opensource@google.com"

/* Define to the full name of this package. */
#define PACKAGE_NAME "glog"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "glog 0.3.5"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "glog"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "0.3.5"

/* How to access the PC from a struct ucontext */
/* #undef PC_FROM_UCONTEXT */

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* The size of 'void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* Define to 1 if you have the ANSI C header files. */
/* #undef STDC_HEADERS */

/* the namespace where STL code like vector<> is defined */
#define STL_NAMESPACE std

/* location of source code */
#define TEST_SRC_DIR "external/com_google_glog"

/* Version number of package */
#define VERSION "0.3.5"

/* Stops putting the code inside the Google namespace */
#define _END_GOOGLE_NAMESPACE_ }

/* Puts following code inside the Google namespace */
#define _START_GOOGLE_NAMESPACE_ namespace google {
"""

genrule(
    name = "gen_ios_arm_config",
    outs = ["config.h.ios_arm"],
    cmd = ("echo '%s' > $(location config.h.ios_arm)" % IOS_ARM_CONFIG),
)
