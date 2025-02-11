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

#ifndef PPOCR_PADDLEOCR_H
#define PPOCR_PADDLEOCR_H

#include <include/utility.h>

namespace PaddleOCR {

class Args;

class PPOCR_API PPOCR {
public:
  explicit PPOCR(Args const &args) noexcept;
  virtual ~PPOCR();

  Args const &args() const noexcept;

  std::vector<std::vector<OCRPredictResult>>
  ocr(const std::vector<cv::Mat> &img_list) noexcept;

  std::vector<OCRPredictResult> ocr(const cv::Mat &img) noexcept {
    return ocr(img, true);
  }

protected:
  void det(const cv::Mat &img,
           std::vector<OCRPredictResult> &ocr_results) noexcept;
  void rec(const std::vector<cv::Mat> &img_list,
           std::vector<OCRPredictResult> &ocr_results) noexcept;
  void cls(const std::vector<cv::Mat> &img_list,
           std::vector<OCRPredictResult> &ocr_results) noexcept;

  std::vector<OCRPredictResult> ocr(const cv::Mat &img, bool and_cls) noexcept;

private:
  struct PPOCR_PRIVATE;
  PPOCR_PRIVATE *pri;
};

} // namespace PaddleOCR

#endif // PPOCR_PADDLEOCR_H
