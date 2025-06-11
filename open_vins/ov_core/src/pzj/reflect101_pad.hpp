#pragma once
#include <opencv2/core/cuda.hpp>

namespace ov_core {
cv::cuda::GpuMat reflect101Pad(const cv::cuda::GpuMat& src,
                               int padY, int padX,
                               cv::cuda::Stream& stream);
} // namespace ov_core
