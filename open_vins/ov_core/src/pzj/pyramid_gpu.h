#pragma once
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

namespace ov_core {

/// 使用 CUDA 构建灰度金字塔；每层都保留 winSize 像素的镜像补边。
/// @param img0   GPU 灰度图（单通道）
/// @param pyr    输出：长度 = levels 的 GpuMat 数组，包含 padding
/// @param levels 金字塔层数（含 level-0）
/// @param winSize Lucas-Kanade 光流窗口大小 (必须与后续 LK 调用保持一致)
/// @param stream  CUDA stream，可缺省
void buildPyramidGPU(const cv::cuda::GpuMat& img0,
                     std::vector<cv::cuda::GpuMat>& pyr,
                     int levels,
                     const cv::Size& winSize,
                     cv::cuda::Stream& stream = cv::cuda::Stream::Null());

} // namespace ov_core
