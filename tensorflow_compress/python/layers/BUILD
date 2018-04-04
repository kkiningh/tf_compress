package(
    default_visibility = [
        "//tensorflow_compress:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "layers",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sparse",
    ],
)

py_library(
    name = "sparse",
    srcs = [
        "sparse.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sparse_test",
    size = "small",
    srcs = ["sparse_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_compress",
    ],
)
