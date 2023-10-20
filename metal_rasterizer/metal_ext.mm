/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The code that registers a PyTorch custom operation.
*/


#include <torch/extension.h>
#include "rasterize_points.h"


// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians_metal", &RasterizeGaussiansMetal);
  m.def("rasterize_gaussians_backward_metal", &RasterizeGaussiansBackwardMetal);
  m.def("mark_visible_metal", &markVisibleMetal);
  m.def("mps_softshrink", &mps_softshrink);
}
