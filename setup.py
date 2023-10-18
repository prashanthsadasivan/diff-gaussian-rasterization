#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

ext_modules = []

if torch.cuda.is_available() == True:
    ext_modules.append(CUDAExtension(
        name="diff_gaussian_rasterization._C",
        sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
        extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
                       )

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    ext_modules.append(CppExtension(
        name="diff_gaussian_rasterization._C",
        sources=[
            "metal_rasterizer/metal_ext.mm"
            #"metal_rasterizer/rasterize_points.cpp", #todo move to metal
            ]
        , extra_cflags=['-std=c++17']))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=ext_modules,
        cmdclass={
            'build_ext': BuildExtension
            }
        )

