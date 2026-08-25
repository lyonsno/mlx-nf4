import os
import sys

from setuptools import setup

from mlx import extension


class CMakeBuild(extension.CMakeBuild):
    """Keep MLX headers, CMake config, and dylib on one Python route."""

    def build_extension(self, ext):
        name = "MLX_NF4_PYTHON_EXECUTABLE"
        previous = os.environ.get(name)
        os.environ[name] = sys.executable
        try:
            super().build_extension(ext)
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


if __name__ == "__main__":
    setup(
        ext_modules=[extension.CMakeExtension("mlx_nf4._ext")],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
    )
