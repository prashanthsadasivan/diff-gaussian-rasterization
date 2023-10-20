/*
  Copyright (C) 2023, Inria
  GRAPHDECO research group, https://team.inria.fr/graphdeco
  All rights reserved.
 
  This software is free for non-commercial, research and evaluation use 
  under the terms of the LICENSE.md file.
 
  For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <memory>
#include "config.h"
#include <fstream>
#include <string>
#include <functional>
#include "metal_ext.h"
#include "rasterize_points.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

void yo(std::string s) {
  std::cout << s << std::endl;
}

torch::Tensor& dispatchSoftShrinkKernel(const torch::Tensor& input, torch::Tensor& output, float lambda) {
  std::cout << "create system default device" << std::endl;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  std::cout << "got device" << std::endl;
  NSError *error = nil;

  // Set the number of threads equal to the number of elements within the input tensor.
  int numThreads = input.numel();

  // Load the custom soft shrink shader.
  std::cout << "new library with source" << std::endl;
  id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                            options:nil
                                                              error:&error];
  std::cout << "new library with source success" << std::endl;
  TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

  std::string kernel_name = std::string("softshrink_kernel_") + (input.scalar_type() == torch::kFloat ? "float" : "half");
  id<MTLFunction> customSoftShrinkFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
  TORCH_CHECK(customSoftShrinkFunction, "Failed to create function state object for ", kernel_name.c_str());
  yo("created function state object");
  // Create a compute pipeline state object for the soft shrink kernel.
  id<MTLComputePipelineState> softShrinkPSO = [device newComputePipelineStateWithFunction:customSoftShrinkFunction error:&error];
  TORCH_CHECK(softShrinkPSO, error.localizedDescription.UTF8String);
  yo("created pipeline state function");

  // Get a reference to the command buffer for the MPS stream.
  id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
  TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");
  yo("created command buffer");


  // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
  dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
  yo("got dispatch queue");

  dispatch_sync(serialQueue, ^(){
      // Start a compute pass.
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");
      yo("created compute command encoder");

      // Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState:softShrinkPSO];
      [computeEncoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
      [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:1];
      [computeEncoder setBytes:&lambda length:sizeof(float) atIndex:2];
      yo("encode the pipeline state object");

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

      // Calculate a thread group size.
      NSUInteger threadGroupSize = softShrinkPSO.maxTotalThreadsPerThreadgroup;
      if (threadGroupSize > numThreads) {
          threadGroupSize = numThreads;
      }
      MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
      yo("mtl size made");

      // Encode the compute command.
      [computeEncoder dispatchThreads:gridSize
                threadsPerThreadgroup:threadgroupSize];
      yo("compute command encoded");

      [computeEncoder endEncoding];
      yo("enconding ended");

      // Commit the work.
      torch::mps::commit();
      yo("work committed");
  });
  return output;
}



std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansMetal(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kMPS);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardMetal(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

// C++ op dispatching the Metal soft shrink shader.
torch::Tensor mps_softshrink(const torch::Tensor &input, float lambda = 0.5) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Check the supported data types for soft shrink.
    TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                input.scalar_type() == torch::kHalf, "Unsupported data type: ", input.scalar_type());

    // Allocate the output, same shape as the input.
    torch::Tensor output = torch::empty_like(input);
    return dispatchSoftShrinkKernel(input, output, lambda);
    //return output;
}

torch::Tensor markVisibleMetal(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  return present;
}
