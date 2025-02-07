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

#include <include/structure_table.h>
#include <include/args.h>
#include <paddle_inference_api.h>

#include <numeric>

namespace PaddleOCR {

StructureTableRecognizer::StructureTableRecognizer(Args const & args) noexcept :
  args_(args),
  mean_({0.485f, 0.456f, 0.406f}),
  scale_({1 / 0.229f, 1 / 0.224f, 1 / 0.225f}),
  is_scale_(true),
  post_processor_(args.table_char_dict_path, args.merge_no_span_structure)
{
  LoadModel(args.table_model_dir);
}

void StructureTableRecognizer::Run(
    const std::vector<cv::Mat> &img_list,
    std::vector<std::vector<std::string>> &structure_html_tags,
    std::vector<float> &structure_scores,
    std::vector<std::vector<std::vector<int>>> &structure_boxes) noexcept {
  size_t img_num = img_list.size();
  for (size_t beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += args_.table_batch_num) {
    // preprocess
    size_t end_img_no = std::min(img_num, beg_img_no + args_.table_batch_num);
    int batch_num = end_img_no - beg_img_no;
    std::vector<cv::Mat> norm_img_batch;
    std::vector<int> width_list;
    std::vector<int> height_list;
    for (size_t ino = beg_img_no; ino < end_img_no; ++ino) {
      cv::Mat srcimg;
      img_list[ino].copyTo(srcimg);
      cv::Mat resize_img;
      cv::Mat pad_img;
      this->resize_op_.Run(srcimg, resize_img, args_.table_max_len);
      this->normalize_op_.Run(resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      this->pad_op_.Run(resize_img, pad_img, args_.table_max_len);
      norm_img_batch.emplace_back(std::move(pad_img));
      width_list.emplace_back(srcimg.cols);
      height_list.emplace_back(srcimg.rows);
    }

    std::vector<float> input(
        batch_num * 3 * args_.table_max_len * args_.table_max_len, 0.0f);
    this->permute_op_.Run(norm_img_batch, input.data());
    // inference.
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape(
        {batch_num, 3, args_.table_max_len, args_.table_max_len});
    input_t->CopyFromCpu(input.data());
    this->predictor_->Run();
    auto output_names = this->predictor_->GetOutputNames();
    auto output_tensor0 = this->predictor_->GetOutputHandle(output_names[0]);
    auto output_tensor1 = this->predictor_->GetOutputHandle(output_names[1]);
    std::vector<int> predict_shape0 = output_tensor0->shape();
    std::vector<int> predict_shape1 = output_tensor1->shape();

    int out_num0 = std::accumulate(predict_shape0.begin(), predict_shape0.end(),
                                   1, std::multiplies<int>());
    int out_num1 = std::accumulate(predict_shape1.begin(), predict_shape1.end(),
                                   1, std::multiplies<int>());
    std::vector<float> loc_preds;
    std::vector<float> structure_probs;
    loc_preds.resize(out_num0);
    structure_probs.resize(out_num1);

    output_tensor0->CopyToCpu(loc_preds.data());
    output_tensor1->CopyToCpu(structure_probs.data());
    // postprocess
    std::vector<std::vector<std::string>> structure_html_tag_batch;
    std::vector<float> structure_score_batch;
    std::vector<std::vector<std::vector<int>>> structure_boxes_batch;
    this->post_processor_.Run(loc_preds, structure_probs, structure_score_batch,
                              predict_shape0, predict_shape1,
                              structure_html_tag_batch, structure_boxes_batch,
                              width_list, height_list);
    for (int m = 0; m < predict_shape0[0]; ++m) {

      structure_html_tag_batch[m].emplace(structure_html_tag_batch[m].begin(),
                                          "<table>");
      structure_html_tag_batch[m].emplace(structure_html_tag_batch[m].begin(),
                                          "<body>");
      structure_html_tag_batch[m].emplace(structure_html_tag_batch[m].begin(),
                                          "<html>");
      structure_html_tag_batch[m].emplace_back("</table>");
      structure_html_tag_batch[m].emplace_back("</body>");
      structure_html_tag_batch[m].emplace_back("</html>");
      structure_html_tags.emplace_back(std::move(structure_html_tag_batch[m]));
      structure_scores.emplace_back(structure_score_batch[m]);
      structure_boxes.emplace_back(std::move(structure_boxes_batch[m]));
    }
  }
}

void StructureTableRecognizer::LoadModel(
    const std::string &model_dir) noexcept {
  paddle_infer::Config config;
  config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");

  if (args_.use_gpu) {
    config.EnableUseGpu(args_.gpu_mem, args_.gpu_id);
    if (args_.use_tensorrt) {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (args_.precision == "fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      if (args_.precision == "int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      }
      config.EnableTensorRtEngine(1 << 20, 10, 3, precision, false, false);
      if (!Utility::PathExists("./trt_table_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_table_shape.txt");
      } else {
        config.EnableTunedTensorRtDynamicShape("./trt_table_shape.txt", true);
      }
    }
  } else {
    config.DisableGpu();
    if (args_.enable_mkldnn) {
      config.EnableMKLDNN();
    } else {
      config.DisableMKLDNN();
    }
    config.SetCpuMathLibraryNumThreads(args_.cpu_threads);
  }

  // false for zero copy tensor
  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = paddle_infer::CreatePredictor(config);
}
} // namespace PaddleOCR
