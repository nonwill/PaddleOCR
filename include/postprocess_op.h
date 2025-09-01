// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PPOCR_POSTPROCESS_OP_HH
#define PPOCR_POSTPROCESS_OP_HH

#include <include/utility.h>

namespace PaddleOCR {

class DBPostProcessor {
public:
  static void GetContourArea(const std::vector<std::vector<float>> &box,
                             float unclip_ratio, float &distance) noexcept;

  static cv::RotatedRect UnClip(const std::vector<std::vector<float>> &box,
                                const float &unclip_ratio) noexcept;

  static float **Mat2Vec(const cv::Mat &mat) noexcept;

  static std::vector<std::vector<int>>
  OrderPointsClockwise(const std::vector<std::vector<int>> &pts) noexcept;

  static std::vector<std::vector<float>>
  GetMiniBoxes(const cv::RotatedRect &box, float &ssid) noexcept;

  static float BoxScoreFast(const std::vector<std::vector<float>> &box_array,
                            const cv::Mat &pred) noexcept;
  static float PolygonScoreAcc(const std::vector<cv::Point> &contour,
                               const cv::Mat &pred) noexcept;

  static std::vector<std::vector<std::vector<int>>>
  BoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
                  const float &box_thresh, const float &det_db_unclip_ratio,
                  const std::string &det_db_score_mode) noexcept;

  static void FilterTagDetRes(std::vector<std::vector<std::vector<int>>> &boxes,
                              float ratio_h, float ratio_w,
                              const cv::Mat &srcimg) noexcept;

private:
  static bool XsortInt(const std::vector<int> &a,
                       const std::vector<int> &b) noexcept;

  static bool XsortFp32(const std::vector<float> &a,
                        const std::vector<float> &b) noexcept;

  static std::vector<std::vector<float>>
  Mat2Vector(const cv::Mat &mat) noexcept;

  static inline int _max(int a, int b) noexcept { return a >= b ? a : b; }

  static inline int _min(int a, int b) noexcept { return a >= b ? b : a; }

  template <class T> inline static T clamp(T x, T min, T max) noexcept {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

  inline static float clampf(float x, float min, float max) noexcept {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }
};

class TablePostProcessor {
public:
  TablePostProcessor(const std::string &label_path,
                     bool merge_no_span_structure = true) noexcept;
  void Run(const std::vector<float> &loc_preds,
           const std::vector<float> &structure_probs,
           std::vector<float> &rec_scores,
           const std::vector<int> &loc_preds_shape,
           const std::vector<int> &structure_probs_shape,
           std::vector<std::vector<std::string>> &rec_html_tag_batch,
           std::vector<std::vector<std::vector<int>>> &rec_boxes_batch,
           const std::vector<int> &width_list,
           const std::vector<int> &height_list) const noexcept;

private:
  static std::vector<std::string>
  gen_label_list(const std::string &label_path,
                 bool merge_no_span_structure) noexcept;

  const std::vector<std::string> label_list_;
  static const std::string end;
  static const std::string beg;
};

class PicodetPostProcessor {
public:
  PicodetPostProcessor(const std::string &label_path,
                       const double score_threshold = 0.4,
                       const double nms_threshold = 0.5,
                       const std::vector<int> &fpn_stride = {8, 16, 32,
                                                             64}) noexcept;
  void Run(std::vector<StructurePredictResult> &results,
           const std::vector<std::vector<float>> &outs,
           const std::vector<int> &ori_shape,
           const std::vector<int> &resize_shape, int eg_max) const noexcept;

  inline size_t fpn_stride_size() const noexcept { return fpn_stride_.size(); }

private:
  StructurePredictResult disPred2Bbox(const std::vector<float> &bbox_pred,
                                      int label, float score, int x, int y,
                                      int stride,
                                      const std::vector<int> &im_shape,
                                      int reg_max) const noexcept;
  void nms(std::vector<StructurePredictResult> &input_boxes,
           float nms_threshold) const noexcept;

  const std::vector<std::string> label_list_;
  const std::vector<int> fpn_stride_;

  const double score_threshold_;
  const double nms_threshold_;
};

} // namespace PaddleOCR

#endif // PPOCR_POSTPROCESS_OP_HH
