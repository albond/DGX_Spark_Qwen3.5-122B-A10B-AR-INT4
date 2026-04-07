"""Build TQ fused decode CUDA extension for SM121 (DGX Spark GB10)."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tq_fused_decode_ext",
    ext_modules=[
        CUDAExtension(
            name="tq_fused_decode_ext",
            sources=["tq_fused_decode.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_121,code=sm_121",
                    "--ptxas-options=-v",  # Show register usage
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
