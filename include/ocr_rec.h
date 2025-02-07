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

#ifndef PPOCR_OCR_REC_HH
#define PPOCR_OCR_REC_HH

#include <include/preprocess_op.h>
#include <include/utility.h>
#include <memory>

namespace paddle_infer {
class Predictor;
}

namespace PaddleOCR {

class Args;

class CRNNRecognizer {
public:
  explicit CRNNRecognizer(Args const & args) noexcept;

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir) noexcept;

  void Run(const std::vector<cv::Mat> &img_list,
           std::vector<std::string> &rec_texts,
           std::vector<float> &rec_text_scores) noexcept;

private:
  static std::vector<std::string> gen_label_list(const std::string &label_path) noexcept;

  Args const & args_;
  const std::vector<std::string> label_list_;
  const std::vector<int> rec_image_shape_;

  const std::vector<float> mean_;
  const std::vector<float> scale_;
  const bool is_scale_;

  std::shared_ptr<paddle_infer::Predictor> predictor_;

  // pre-process
  CrnnResizeImg resize_op_;
  Normalize normalize_op_;
  PermuteBatch permute_op_;

}; // class CrnnRecognizer

} // namespace PaddleOCR

#endif // PPOCR_OCR_REC_HH
