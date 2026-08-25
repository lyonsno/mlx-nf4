from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        ext_modules=[extension.CMakeExtension("mlx_nf4._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        zip_safe=False,
    )
