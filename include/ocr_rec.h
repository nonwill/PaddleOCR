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

class CRNNRecognizer {
public:
  explicit CRNNRecognizer(const std::string &model_dir,
                          const std::string &label_path,
                          const bool &use_gpu = false,
                          const int &gpu_id = 0, const int &gpu_mem = 4000,
                          const int &cpu_math_library_num_threads = 4,
                          const bool &use_mkldnn = false,
                          const bool &use_tensorrt = false,
                          const std::string &precision = "fp32",
                          const int &rec_batch_num = 6,
                          const int &rec_img_h = 32,
                          const int &rec_img_w = 320) noexcept :
    use_gpu_(use_gpu), gpu_id_(gpu_id), gpu_mem_(gpu_mem),
    cpu_math_library_num_threads_(cpu_math_library_num_threads),
    use_mkldnn_(use_mkldnn), use_tensorrt_(use_tensorrt),
    precision_(precision), rec_batch_num_(rec_batch_num),
    rec_img_h_(rec_img_h), rec_img_w_(rec_img_w)
  {
    std::vector<int> rec_image_shape = {3, rec_img_h, rec_img_w};
    this->rec_image_shape_ = rec_image_shape;

    this->label_list_ = Utility::ReadDict(label_path);
    this->label_list_.emplace(this->label_list_.begin(),
                              "#"); // blank char for ctc
    this->label_list_.emplace_back(" ");

    LoadModel(model_dir);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir) noexcept;

  void Run(const std::vector<cv::Mat> &img_list,
           std::vector<std::string> &rec_texts,
           std::vector<float> &rec_text_scores,
           std::vector<double> &times) noexcept;

private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;

  std::vector<std::string> label_list_;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;
  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";
  int rec_batch_num_ = 6;
  int rec_img_h_ = 32;
  int rec_img_w_ = 320;
  std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};
  // pre-process
  CrnnResizeImg resize_op_;
  Normalize normalize_op_;
  PermuteBatch permute_op_;

}; // class CrnnRecognizer

} // namespace PaddleOCR

#endif // PPOCR_OCR_REC_HH
