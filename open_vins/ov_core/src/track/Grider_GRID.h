

#ifndef OV_CORE_GRIDER_GRID_H
#define OV_CORE_GRIDER_GRID_H

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utils/opencv_lambda_body.h"

#include <opencv2/cudafeatures2d.hpp>   // GPU 版 FastFeatureDetector
#include <opencv2/cudaimgproc.hpp>      // 如果后续需要做 GPU 级预处理
#include <opencv2/core/cuda_stream_accessor.hpp> // 可选：访问底层 cudaStream_t

using cv::cuda::GpuMat;
using cv::cuda::Stream;

namespace ov_core {

/**
 * @brief Extracts FAST features in a grid pattern.
 *
 * As compared to just extracting fast features over the entire image,
 * we want to have as uniform of extractions as possible over the image plane.
 * Thus we split the image into a bunch of small grids, and extract points in each.
 * We then pick enough top points in each grid so that we have the total number of desired points.
 */
class Grider_GRID {
public:
  /**
   * @brief Compare keypoints based on their response value.
   * @param first First keypoint
   * @param second Second keypoint
   *
   * We want to have the keypoints with the highest values!
   * See: https://stackoverflow.com/a/10910921
   */
  static bool compare_response(const cv::KeyPoint &first, const cv::KeyPoint &second) {
    return first.response > second.response;
  }

  /**
   * @brief This function will perform grid extraction using FAST.
   * @param img Image we will do FAST extraction on
   * @param mask Region of the image we do not want to extract features in (255 = do not detect features)
   * @param valid_locs Valid 2d grid locations we will extract in (instead of the whole image)
   * @param pts vector of extracted points we will return
   * @param num_features max number of features we want to extract
   * @param grid_x size of grid in the x-direction / u-direction
   * @param grid_y size of grid in the y-direction / v-direction
   * @param threshold FAST threshold parameter (10 is a good value normally)
   * @param nonmaxSuppression if FAST should perform non-max suppression (true normally)
   *
   * Given a specified grid size, this will try to extract fast features from each grid.
   * It will then return the best from each grid in the return vector.
   */
  static void perform_griding(const cv::Mat &img,
                              const cv::Mat &mask,
                              const std::vector<std::pair<int,int>> &valid_locs,
                              std::vector<cv::KeyPoint> &pts,
                              int num_features,
                              int grid_x, int grid_y,
                              int threshold,
                              bool nonmaxSuppression)
  {
    /* ---------- 0. quick exit ---------- */
    if (valid_locs.empty()) {
      return;
    }

    /* ---------- 1. adjust grid ---------- */
    if (num_features < grid_x * grid_y) {
      double r = (double)grid_x / (double)grid_y;
      grid_y   = (int)std::ceil(std::sqrt(num_features / r));
      grid_x   = (int)std::ceil(grid_y * r);
    }
    int num_features_grid = (int)((double)num_features / (grid_x * grid_y)) + 1;

    // 采用向上取整来计算每格大小，防止边界像素被丢弃
    int size_x = (img.cols + grid_x - 1) / grid_x;
    int size_y = (img.rows + grid_y - 1) / grid_y;
    CV_Assert(size_x > 0 && size_y > 0);

    /* ---------- 2. prepare grayscale image ---------- */
    cv::Mat gray;
    if (img.channels() == 3) {
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.type() == CV_8UC1) {
      gray = img;
    } else {
      // 如果输入不是 8-bit 单通道或 3 通道，先把它归一化到 8UC1
      double minv, maxv;
      cv::minMaxLoc(img, &minv, &maxv);
      img.convertTo(gray, CV_8UC1, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    }
    CV_Assert(gray.type() == CV_8UC1);

    /* ---------- 3. upload grayscale to GPU ---------- */
    cv::cuda::GpuMat d_img(gray);

    /* ---------- 4. GPU FAST (先不传掩膜，仅做最简检测) ---------- */
    // 如果你后面需要恢复掩膜，可以把 detectAsync 的第三个参数换成 d_mask
    auto fast_gpu = cv::cuda::FastFeatureDetector::create(
      // 先把阈值调小一半试试，如果确实能检测到再改回
      std::max(threshold / 2, 5),
      nonmaxSuppression,
      cv::FastFeatureDetector::TYPE_9_16,
      20000  // maxKeyPoints
    );

    cv::cuda::Stream stream;
    cv::cuda::GpuMat d_kps;               // GPU 上存储 keypoints 的缓冲
    std::vector<cv::KeyPoint> kps_host;   // 拷回 CPU 的 keypoints

    // 第三个参数改成空，即先不传入掩膜
    fast_gpu->detectAsync(d_img, d_kps, cv::cuda::GpuMat(), stream);
    stream.waitForCompletion();
    fast_gpu->convert(d_kps, kps_host);

    if (kps_host.empty()) {
      // GPU FAST 没检测到任何点，说明阈值还是过高，或图像不适合 CUDA FAST
      // 这时就只能退回 CPU 版，或者再调更低阈值
      return;
    }

    /* ---------- 5. bucket by grid ---------- */
    std::vector<std::vector<cv::KeyPoint>> buckets(valid_locs.size());
    for (const auto &kp : kps_host) {
      int gx = kp.pt.x / size_x;
      int gy = kp.pt.y / size_y;
      gx = std::min(grid_x - 1, std::max(0, gx));
      gy = std::min(grid_y - 1, std::max(0, gy));

      auto it = std::find(valid_locs.begin(), valid_locs.end(), std::make_pair(gx, gy));
      if (it == valid_locs.end()) continue;

      int idx = static_cast<int>(std::distance(valid_locs.begin(), it));
      buckets[idx].push_back(kp);
    }

    /* ---------- 6. collect top keypoints per bucket ---------- */
    pts.clear();
    for (auto &vec : buckets) {
      if (vec.empty()) continue;
      std::sort(vec.begin(), vec.end(),
                [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                  return a.response > b.response;
                });
      if (vec.size() > (size_t)num_features_grid) {
        vec.resize(num_features_grid);
      }
      pts.insert(pts.end(), vec.begin(), vec.end());
    }

    if (pts.empty()) {
      // 全部 bucket 最终都没点，可能 valid_locs 与实际检测区域不匹配
      return;
    }

    /* ---------- 7. sub-pixel refinement ---------- */
    std::vector<cv::Point2f> pts_refined;
    pts_refined.reserve(pts.size());
    for (auto &k : pts) {
      pts_refined.emplace_back(k.pt);
    }

    cv::cornerSubPix(
      gray,
      pts_refined,
      cv::Size(5, 5),
      cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.001)
    );

    for (size_t i = 0; i < pts.size(); ++i) {
      pts[i].pt = pts_refined[i];
    }
  }
};  // class Grider_GRID

}  // namespace ov_core

#endif  // OV_CORE_GRIDER_GRID_H
