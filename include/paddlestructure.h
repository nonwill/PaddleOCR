// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef PPOCR_PADDLESTRUCTURE_H
#define PPOCR_PADDLESTRUCTURE_H

#include <include/paddleocr.h>

namespace PaddleOCR {

class StructureTableRecognizer;
class StructureLayoutRecognizer;

class PPOCR_API PaddleStructure : public PPOCR {
public:
#ifdef PPOCR_gflags_ENABLED
  explicit PaddleStructure() noexcept;
#else
  explicit PaddleStructure(Args const & args) noexcept;
#endif
  ~PaddleStructure();

  std::vector<StructurePredictResult> structure(const cv::Mat &img,
                                                bool layout = false,
                                                bool table = true,
                                                bool ocr = false) noexcept;

#ifdef PPOCR_benchmark_ENABLED
  void reset_timer() noexcept;
  void benchmark_log(int img_num) noexcept;
#endif

private:
#ifdef PPOCR_benchmark_ENABLED
  std::vector<double> time_info_table = {0, 0, 0};
  std::vector<double> time_info_layout = {0, 0, 0};
#endif

  StructureTableRecognizer * table_model;
  StructureLayoutRecognizer * layout_model;

  void layout(const cv::Mat &img,
              std::vector<StructurePredictResult> &structure_result) noexcept;

  void table(const cv::Mat &img,
             StructurePredictResult &structure_result) noexcept;

  std::string rebuild_table(const std::vector<std::string> &rec_html_tags,
                            const std::vector<std::vector<int>> &rec_boxes,
                            std::vector<OCRPredictResult> &ocr_result) noexcept;

  float dis(const std::vector<int> &box1,
            const std::vector<int> &box2) noexcept;

  static bool comparison_dis(const std::vector<float> &dis1,
                             const std::vector<float> &dis2) noexcept {
    if (dis1[1] < dis2[1]) {
      return true;
    } else if (dis1[1] == dis2[1]) {
      return dis1[0] < dis2[0];
    } else {
      return false;
    }
  }
};

} // namespace PaddleOCR

#endif // PPOCR_PADDLESTRUCTURE_H
