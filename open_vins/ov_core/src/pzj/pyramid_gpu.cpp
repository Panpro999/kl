#include "pzj/pyramid_gpu.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>      // cv::copyMakeBorder
#include "pzj/reflect101_pad.hpp"

namespace ov_core {


/* ---- CUDA 版 buildOpticalFlowPyramid ---- */
void buildPyramidGPU(const cv::cuda::GpuMat& img0,
                     std::vector<cv::cuda::GpuMat>& pyr,
                     int levels,
                     const cv::Size& winSize,
                     cv::cuda::Stream& stream)
{
    CV_Assert(levels >= 1);
    pyr.resize(levels);

    const int pad_x = winSize.width;   // ★ 与 CPU 版保持一致：整窗口大小
    const int pad_y = winSize.height;

    /* ---------- level-0 ---------- */
    pyr[0] = ov_core::reflect101Pad(img0,
                                    pad_y, pad_x,
                                    stream);

    /* ---------- 后续层 ---------- */
    cv::Size roi_sz(img0.cols, img0.rows);   // 当前中心 ROI 尺寸

    for (int l = 1; l < levels; ++l)
    {
        /* ① 取上一层中心 ROI（去掉 padding） */
        cv::cuda::GpuMat prev_roi =
            pyr[l - 1](cv::Rect(pad_x, pad_y, roi_sz.width, roi_sz.height));

        /* ② 对中心 ROI 下采样（CUDA 版不需要 Size 参数） */
        cv::Size half((roi_sz.width + 1) >> 1, (roi_sz.height + 1) >> 1);
        cv::cuda::GpuMat half_roi;
        cv::cuda::pyrDown(prev_roi, half_roi, stream);

        /* ③ 对结果再补边 */
      /* ③ 对结果再补边（GPU Reflect-101） */
       pyr[l] = ov_core::reflect101Pad(half_roi,
                                       pad_y, pad_x,
                                       stream);

        roi_sz = half;  // 更新下一轮 ROI 尺寸
    }
}

} // namespace ov_core
