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

#ifndef PPOCR_OCR_CLS_HH
#define PPOCR_OCR_CLS_HH

#include <include/preprocess_op.h>
#include <include/utility.h>
#include <memory>

namespace paddle_infer { class Predictor; }

namespace PaddleOCR {

class Classifier {
public:
  explicit Classifier(const std::string &model_dir,
                      const bool &use_gpu = false,
                      const int &gpu_id = 0,
                      const int &gpu_mem = 4000,
                      const int &cpu_math_library_num_threads = 4,
                      const bool &use_mkldnn = false,
                      const double &cls_thresh = 0.9,
                      const bool &use_tensorrt = false,
                      const std::string &precision = "fp32",
                      const int &cls_batch_num = 1) noexcept :
    use_gpu_(use_gpu), gpu_id_(gpu_id), gpu_mem_(gpu_mem),
    cpu_math_library_num_threads_(cpu_math_library_num_threads),
    use_mkldnn_(use_mkldnn), cls_thresh(cls_thresh),
    use_tensorrt_(use_tensorrt), precision_(precision),
    cls_batch_num_(cls_batch_num)
  {
    LoadModel(model_dir);
  }

  inline double cls_thresh_val() const noexcept { return cls_thresh; }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir) noexcept;

  void Run(const std::vector<cv::Mat> &img_list, std::vector<int> &cls_labels,
           std::vector<float> &cls_scores, std::vector<double> &times) noexcept;

private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;
  double cls_thresh = 0.9;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;
  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";
  int cls_batch_num_ = 1;
  // pre-process
  ClsResizeImg resize_op_;
  Normalize normalize_op_;
  PermuteBatch permute_op_;

}; // class Classifier

} // namespace PaddleOCR

#endif // PPOCR_OCR_CLS_HH
