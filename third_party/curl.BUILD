# Copyright 2026 The MediaPipe Authors.
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

genrule(
    name = "nghttp2_h",
    srcs = [
        "@nghttp2_archive//:nghttp2_h",
    ],
    outs = ["lib/nghttp2/nghttp2.h"],
    cmd = "cp -v $< $@",
)

cc_library(
    name = "curl",
    srcs = glob(["lib/**/*.c"]),
    hdrs = glob(["lib/**/*.h"]) + glob(["include/curl/**/*.h"]) + [
        "lib/curl_config.h",
    ],
    copts = [
        "-D_GNU_SOURCE",
        "-DALLOW_RENEG",
        "-DHAVE_CONFIG_H",
        "-DCURL_DISABLE_NTLM",  # turning it off in configure is not enough.
        "-DHAVE_BORINGSSL",
        "-DBUILDING_LIBCURL",
        "-DCURL_ENABLE_SMTP",
        "-DOPENSSL_IS_BORINGSSL",
        "-DCURL_DISABLE_LIBCURL_OPTION",
        "-DCURL_MAX_WRITE_SIZE=65536",
    ],
    includes = [
        "include",
        "include/curl",
        "lib",
    ],
    linkopts = ["-lrt"],
    visibility = ["//visibility:public"],
    deps = [
        "@boringssl//:crypto",
        "@boringssl//:ssl",
    ],
)

sizeof_short = select({
    "@platforms//cpu:aarch64": "2",
    "//conditions:default": "2",
})

sizeof_int = select({
    "@platforms//cpu:aarch64": "4",
    "//conditions:default": "4",
})

sizeof_long = select({
    "@platforms//cpu:aarch64": "4",
    "//conditions:default": "8",
})

sizeof_ptr = select({
    "@platforms//cpu:aarch64": "4",
    "//conditions:default": "8",
})

sizeof_off_t = sizeof_long

sizeof_size_t = sizeof_long

sizeof_socklen_t = "4"

sizeof_time_t = sizeof_long

genrule(
    name = "generate_curl_config_h",
    srcs = ["lib/curl_config.h.cmake"],
    outs = ["lib/curl_config.h"],
    cmd = ("sed -e 's|#cmakedefine \\(CURL_CA_BUNDLE\\) .*$$|#define \\1 \"/etc/ssl/certs/ca-certificates.crt\"|'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_DICT\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_FILE\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_FTP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_GOPHER\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_IMAP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_LDAP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_LDAPS\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_LIBCURL_OPTION\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_POP3\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_TELNET\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_DISABLE_TFTP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(CURL_EXTERN_SYMBOL\\) .*$$/#define \\1 __attribute__ ((__visibility__ (\"default\")))/'" +
           "    -e 's/#cmakedefine \\(ENABLE_IPV6\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(GETHOSTNAME_TYPE_ARG2\\) .*$$/#define \\1 size_t/'" +
           "    -e 's/#cmakedefine \\(GETNAMEINFO_QUAL_ARG1\\) .*$$/#define \\1 const/'" +
           "    -e 's/#cmakedefine \\(GETNAMEINFO_TYPE_ARG1\\) .*$$/#define \\1 struct sockaddr */'" +
           "    -e 's/#cmakedefine \\(GETNAMEINFO_TYPE_ARG2\\) .*$$/#define \\1 socklen_t/'" +
           "    -e 's/#cmakedefine \\(GETNAMEINFO_TYPE_ARG46\\) .*$$/#define \\1 socklen_t/'" +
           "    -e 's/#cmakedefine \\(GETNAMEINFO_TYPE_ARG7\\) .*$$/#define \\1 int/'" +
           "    -e 's/#cmakedefine \\(GETSERVBYPORT_R_ARGS\\) .*$$/#define \\1 6/'" +
           "    -e 's/#cmakedefine \\(GETSERVBYPORT_R_BUFSIZE\\) .*$$/#define \\1 4096/'" +
           "    -e 's/#cmakedefine \\(HAVE_ALARM\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_ALLOCA_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_ARPA_INET_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_ARPA_TFTP_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_ASSERT_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_BASENAME\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_BOOL_T\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_BORINGSSL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_CLOCK_GETTIME_MONOTONIC\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_CONNECT\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_CRYPTO_CLEANUP_ALL_EX_DATA\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_DES_SET_ODD_PARITY\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_DLFCN_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_ERRNO_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FCNTL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FCNTL_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FCNTL_O_NONBLOCK\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FDOPEN\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FORK\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FREEADDRINFO\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FREEIFADDRS\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FSETXATTR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FSETXATTR_5\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_FTRUNCATE\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GAI_STRERROR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETADDRINFO\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETADDRINFO_THREADSAFE\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETEUID\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETHOSTBYADDR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETHOSTBYADDR_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETHOSTBYADDR_R_8\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETHOSTBYNAME\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETHOSTBYNAME_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETHOSTBYNAME_R_6\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETHOSTNAME\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETIFADDRS\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETNAMEINFO\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETPPID\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETPROTOBYNAME\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETPWUID\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETPWUID_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETRLIMIT\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETSERVBYPORT_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GETTIMEOFDAY\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_GMTIME_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_IFADDRS_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_IF_NAMETOINDEX\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_INET_ADDR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_INET_NTOP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_INET_PTON\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_INTTYPES_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_IOCTL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_IOCTL_FIONBIO\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_IOCTL_SIOCGIFADDR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_LIBGEN_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_LIBSSL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_LIMITS_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_LL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_LOCALE_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_LOCALTIME_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_LONGLONG\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_MALLOC_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_MEMORY_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_MSG_NOSIGNAL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_NETDB_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_NETINET_IN_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_NETINET_TCP_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_NET_IF_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_OPENSSL_CRYPTO_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_OPENSSL_ERR_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_OPENSSL_PEM_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_OPENSSL_PKCS12_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_OPENSSL_RSA_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_OPENSSL_SSL_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_OPENSSL_X509_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_PERROR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_PIPE\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_POLL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_POLL_FINE\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_POLL_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_POSIX_STRERROR_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_PWD_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_RAND_EGD\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_RAND_STATUS\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_RECV\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SELECT\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SEND\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SETJMP_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SETLOCALE\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SETRLIMIT\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SETSOCKOPT\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SGTTY_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SIGACTION\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SIGINTERRUPT\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SIGNAL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SIGNAL_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SIGSETJMP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SIG_ATOMIC_T\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SOCKET\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SOCKETPAIR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SSLV2_CLIENT_METHOD\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SSL_GET_SHUTDOWN\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STDBOOL_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STDINT_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STDIO_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STDLIB_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRCASECMP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRDUP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRERROR_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRINGS_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRING_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRNCASECMP\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRSTR\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRTOK_R\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRTOLL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRUCT_SOCKADDR_STORAGE\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_STRUCT_TIMEVAL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_IOCTL_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_PARAM_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_POLL_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_RESOURCE_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_SELECT_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_SOCKET_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_STAT_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_TIME_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_TYPES_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_UIO_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_UN_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_WAIT_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_SYS_XATTR_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_TERMIOS_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_TERMIO_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_TIME_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_UNAME\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_UNISTD_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_UTIME\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_UTIME_H\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_VARIADIC_MACROS_C99\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_VARIADIC_MACROS_GCC\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_WRITABLE_ARGV\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_WRITEV\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(HAVE_ZLIB_H\\) .*$$/#define \\1 1/'" +
           "    -e 's|#cmakedefine \\(LT_OBJDIR\\) .*$$|#define \\1 \".libs/\"|'" +
           "    -e 's/#cmakedefine \\(OS\\) .*$$/#define \\1 \"k8-unknown-linux-gnu\"/'" +
           "    -e 's/#cmakedefine \\(PACKAGE\\) .*$$/#define \\1 \"curl\"/'" +
           "    -e 's|#cmakedefine \\(PACKAGE_BUGREPORT\\) .*$$|#define \\1 \"a suitable curl mailing list: http://curl.haxx.se/mail/\"|'" +
           "    -e 's/#cmakedefine \\(PACKAGE_NAME\\) .*$$/#define \\1 \"curl\"/'" +
           "    -e 's/#cmakedefine \\(PACKAGE_STRING\\) .*$$/#define \\1 \"curl -\"/'" +
           "    -e 's/#cmakedefine \\(PACKAGE_TARNAME\\) .*$$/#define \\1 \"curl\"/'" +
           "    -e 's/#cmakedefine \\(PACKAGE_URL\\) .*$$/#define \\1 \"\"/'" +
           "    -e 's/#cmakedefine \\(PACKAGE_VERSION\\) .*$$/#define \\1 \"-\"/'" +
           "    -e 's|#cmakedefine \\(RANDOM_FILE\\) .*$$|#define \\1 \"/dev/urandom\"|'" +
           "    -e 's/#cmakedefine \\(RECV_TYPE_ARG1\\) .*$$/#define \\1 int/'" +
           "    -e 's/#cmakedefine \\(RECV_TYPE_ARG2\\) .*$$/#define \\1 void */'" +
           "    -e 's/#cmakedefine \\(RECV_TYPE_ARG3\\) .*$$/#define \\1 size_t/'" +
           "    -e 's/#cmakedefine \\(RECV_TYPE_ARG4\\) .*$$/#define \\1 int/'" +
           "    -e 's/#cmakedefine \\(RECV_TYPE_RETV\\) .*$$/#define \\1 ssize_t/'" +
           "    -e 's/#cmakedefine \\(RETSIGTYPE\\) .*$$/#define \\1 void/'" +
           "    -e 's/#cmakedefine \\(SELECT_QUAL_ARG5\\) .*$$/#define \\1/'" +
           "    -e 's/#cmakedefine \\(SELECT_TYPE_ARG1\\) .*$$/#define \\1 int/'" +
           "    -e 's/#cmakedefine \\(SELECT_TYPE_ARG234\\) .*$$/#define \\1 fd_set */'" +
           "    -e 's/#cmakedefine \\(SELECT_TYPE_ARG5\\) .*$$/#define \\1 struct timeval */'" +
           "    -e 's/#cmakedefine \\(SELECT_TYPE_RETV\\) .*$$/#define \\1 int/'" +
           "    -e 's/#cmakedefine \\(SEND_QUAL_ARG2\\) .*$$/#define \\1 const/'" +
           "    -e 's/#cmakedefine \\(SEND_TYPE_ARG1\\) .*$$/#define \\1 int/'" +
           "    -e 's/#cmakedefine \\(SEND_TYPE_ARG2\\) .*$$/#define \\1 void */'" +
           "    -e 's/#cmakedefine \\(SEND_TYPE_ARG3\\) .*$$/#define \\1 size_t/'" +
           "    -e 's/#cmakedefine \\(SEND_TYPE_ARG4\\) .*$$/#define \\1 int/'" +
           "    -e 's/#cmakedefine \\(SEND_TYPE_RETV\\) .*$$/#define \\1 ssize_t/'" +
           "    -e 's/\\$${\\(SIZEOF_INT\\)_CODE}$$/#define \\1 " + sizeof_int + "/'" +
           "    -e 's/\\$${\\(SIZEOF_LONG\\)_CODE}$$/#define \\1 " + sizeof_long + "/'" +
           "    -e 's/\\$${\\(SIZEOF_LONG_LONG\\)_CODE}$$/#define \\1 " + sizeof_long + "/'" +
           "    -e 's/\\$${\\(SIZEOF_OFF_T\\)_CODE}$$/#define \\1 " + sizeof_off_t + "/'" +
           "    -e 's/\\$${\\(SIZEOF_CURL_OFF_T\\)_CODE}$$/#define \\1 " + sizeof_off_t + "/'" +
           "    -e 's/\\$${\\(SIZEOF_CURL_SOCKET_T\\)_CODE}$$/#define \\1 " + sizeof_off_t + "/'" +
           "    -e 's/\\$${\\(SIZEOF_SIZE_T\\)_CODE}$$/#define \\1 " + sizeof_size_t + "/'" +
           "    -e 's/\\$${\\(SIZEOF_TIME_T\\)_CODE}$$/#define \\1 " + sizeof_time_t + "/'" +
           "    -e 's/#cmakedefine \\(STDC_HEADERS\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(STRERROR_R_TYPE_ARG3\\) .*$$/#define \\1 size_t/'" +
           "    -e 's/#cmakedefine \\(TIME_WITH_SYS_TIME\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(USE_OPENSSL\\) .*$$/#define \\1 1/'" +
           "    -e 's/#cmakedefine \\(VERSION\\) .*$$/#define \\1 \"-\"/'" +
           "    -e 's/#cmakedefine .*$$//'" +
           "  $< > $@"),
)
