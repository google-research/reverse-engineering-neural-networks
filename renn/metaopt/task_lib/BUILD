py_library(
    name = "regression",
    srcs = ["regression.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/jax",
    ],
)

py_library(
    name = "classification",
    srcs = ["classification.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jax:stax",
        "//third_party/py/tensorflow_datasets",
    ],
)

py_library(
    name = "quadratic",
    srcs = ["quadratic.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/jax",
    ],
)
