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

#include <include/args.h>
#include <include/ocr_cls.h>
#include <include/preprocess_op.h>
#include <paddle_inference_api.h>

#include <numeric>

namespace PaddleOCR {

Classifier::Classifier(Args const &args) noexcept
    : args_(args), mean_(3, 0.5f), scale_(3, 1.0 / 0.5f), is_scale_(true) {
  LoadModel(args_.cls_model_dir);
}

void Classifier::Run(const std::vector<cv::Mat> &img_list,
                     std::vector<int> &cls_labels,
                     std::vector<float> &cls_scores) noexcept {
  int img_num = img_list.size();
  std::vector<int> cls_image_shape = {3, 48, 192};
  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += args_.cls_batch_num) {
    int end_img_no = std::min(img_num, beg_img_no + args_.cls_batch_num);
    int batch_num = end_img_no - beg_img_no;
    // preprocess
    std::vector<cv::Mat> norm_img_batch;
    for (int ino = beg_img_no; ino < end_img_no; ++ino) {
      cv::Mat srcimg;
      img_list[ino].copyTo(srcimg);
      cv::Mat resize_img;
      ClsResizeImg::Run(srcimg, resize_img, args_.use_tensorrt,
                        cls_image_shape);

      Normalize::Run(resize_img, mean_, scale_, is_scale_);
      if (resize_img.cols < cls_image_shape[2]) {
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                           cls_image_shape[2] - resize_img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      }
      norm_img_batch.emplace_back(std::move(resize_img));
    }
    std::vector<float> input(batch_num * cls_image_shape[0] *
                                 cls_image_shape[1] * cls_image_shape[2],
                             0.0f);
    PermuteBatch::Run(norm_img_batch, input.data());

    // inference.
    auto input_names = predictor_->GetInputNames();
    auto input_t = predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape({batch_num, cls_image_shape[0], cls_image_shape[1],
                      cls_image_shape[2]});
    input_t->CopyFromCpu(input.data());
    predictor_->Run();

    std::vector<float> predict_batch;
    auto output_names = predictor_->GetOutputNames();
    auto output_t = predictor_->GetOutputHandle(output_names[0]);
    auto predict_shape = output_t->shape();

    int out_num = std::accumulate(predict_shape.begin(), predict_shape.end(), 1,
                                  std::multiplies<int>());
    predict_batch.resize(out_num);

    output_t->CopyToCpu(predict_batch.data());

    // postprocess
    for (int batch_idx = 0; batch_idx < predict_shape[0]; ++batch_idx) {
      int label = int(
          Utility::argmax(&predict_batch[batch_idx * predict_shape[1]],
                          &predict_batch[(batch_idx + 1) * predict_shape[1]]));
      float score = float(*std::max_element(
          &predict_batch[batch_idx * predict_shape[1]],
          &predict_batch[(batch_idx + 1) * predict_shape[1]]));
      cls_labels[beg_img_no + batch_idx] = label;
      cls_scores[beg_img_no + batch_idx] = score;
    }
  }
}

void Classifier::LoadModel(const std::string &model_dir) noexcept {
  paddle_infer::Config config;
  bool json_model = false;
  std::string model_file_path, param_file_path;
  char const *model_variants[8] = {"/inference.json",    "/inference.pdiparams",
                                   "/model.json",        "/model.pdiparams",
                                   "/inference.pdmodel", "/inference.pdiparams",
                                   "/model.pdmodel",     "/model.pdiparams"};
  for (int i = 0; i < 8; i += 2) {
    std::string model_file = model_dir + model_variants[i];
    if (Utility::PathExists(model_file)) {
      model_file_path = std::move(model_file);
      param_file_path = model_dir + model_variants[i + 1];
      json_model = (i == 0 || i == 2);
      break;
    }
  }
  if (model_file_path.empty()) {
    fprintf(stderr, "[ERROR] No valid model file found in %s\n",
            model_dir.c_str());
    fflush(stderr);
    return;
  }
  config.SetModel(model_file_path, param_file_path);

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
      if (!Utility::PathExists("./trt_cls_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_cls_shape.txt");
      } else {
        config.EnableTunedTensorRtDynamicShape("./trt_cls_shape.txt", true);
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
    if (json_model) {
      config.EnableNewIR();
      config.EnableNewExecutor();
    }
  }

  // false for zero copy tensor
  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  predictor_ = paddle_infer::CreatePredictor(config);
}
} // namespace PaddleOCR
