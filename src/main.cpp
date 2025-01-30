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
#ifdef USING_PPOCR_CPP_API
#include <paddlestructure.h>
#include <args.h>
#include "opencv2/imgcodecs.hpp"
#else
#include <ppocr_c.h>
#endif

#include <iostream>
#include <vector>

#ifdef USING_PPOCR_CPP_API
using namespace PaddleOCR;

#ifdef PPOCR_gflags_ENABLED
void check_params() {
  if (FLAGS_det) {
    if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_rec) {
    std::cout
        << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
           "if you are using recognition model with PP-OCRv2 or an older "
           "version, "
           "please set --rec_image_shape='3,32,320"
        << std::endl;
    if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[rec]: ./ppocr "
                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_cls && FLAGS_use_angle_cls) {
    if (FLAGS_cls_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[cls]: ./ppocr "
                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_table) {
    if (FLAGS_table_model_dir.empty() || FLAGS_det_model_dir.empty() ||
        FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[table]: ./ppocr "
                << "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--table_model_dir=/PATH/TO/TABLE_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_layout) {
    if (FLAGS_layout_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[layout]: ./ppocr "
                << "--layout_model_dir=/PATH/TO/LAYOUT_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
      FLAGS_precision != "int8") {
    std::cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    exit(1);
  }
}
#else
void check_params( Args const & args ) {
  if (args.det) {
    if (args.det_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (args.rec) {
    std::cout
        << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
           "if you are using recognition model with PP-OCRv2 or an older "
           "version, "
           "please set --rec_image_shape='3,32,320"
        << std::endl;
    if (args.rec_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[rec]: ./ppocr "
                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (args.cls && args.use_angle_cls) {
    if (args.cls_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[cls]: ./ppocr "
                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (args.table) {
    if (args.table_model_dir.empty() || args.det_model_dir.empty() ||
        args.rec_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[table]: ./ppocr "
                << "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--table_model_dir=/PATH/TO/TABLE_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (args.layout) {
    if (args.layout_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[layout]: ./ppocr "
                << "--layout_model_dir=/PATH/TO/LAYOUT_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (args.precision != "fp32" && args.precision != "fp16" &&
      args.precision != "int8") {
    std::cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    exit(1);
  }
}
#endif

#ifdef PPOCR_gflags_ENABLED
void ocr(std::vector<cv::String> &cv_all_img_names) {
  PPOCR ocr;
#else
void ocr(std::vector<cv::String> &cv_all_img_names, Args const & args) {
  PPOCR ocr(args);
#endif

#ifdef PPOCR_benchmark_ENABLED
  if (FLAGS_benchmark) {
    ocr.reset_timer();
  }
#endif

  std::vector<cv::Mat> img_list;
  std::vector<cv::String> img_names;
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }
    img_list.emplace_back(img);
    img_names.emplace_back(cv_all_img_names[i]);
  }

#ifdef PPOCR_gflags_ENABLED
  std::vector<std::vector<OCRPredictResult>> ocr_results =
      ocr.ocr(img_list, FLAGS_det, FLAGS_rec, FLAGS_cls);

  for (int i = 0; i < img_names.size(); ++i) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    Utility::print_result(ocr_results[i]);
    if (FLAGS_visualize && FLAGS_det) {
      std::string file_name = Utility::basename(img_names[i]);
      cv::Mat srcimg = img_list[i];
      Utility::VisualizeBboxes(srcimg, ocr_results[i],
                               FLAGS_output + "/" + file_name);
    }
  }
#else
  std::vector<std::vector<OCRPredictResult>> ocr_results =
      ocr.ocr(img_list, args.det, args.rec, args.cls);

  for (int i = 0; i < img_names.size(); ++i) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    Utility::print_result(ocr_results[i]);
    if (args.visualize && args.det) {
      std::string file_name = Utility::basename(img_names[i]);
      cv::Mat srcimg = img_list[i];
      Utility::VisualizeBboxes(srcimg, ocr_results[i],
                               args.output + "/" + file_name);
    }
  }
#endif

#ifdef PPOCR_benchmark_ENABLED
  if (FLAGS_benchmark) {
    ocr.benchmark_log(cv_all_img_names.size());
  }
#endif
}

#ifdef PPOCR_gflags_ENABLED
void structure(std::vector<cv::String> &cv_all_img_names) {
  PaddleOCR::PaddleStructure engine;
#else
void structure(std::vector<cv::String> &cv_all_img_names, Args const & args) {
  PaddleOCR::PaddleStructure engine(args);
#endif

#ifdef PPOCR_benchmark_ENABLED
  if (FLAGS_benchmark) {
    engine.reset_timer();
  }
#endif

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }

#ifdef PPOCR_gflags_ENABLED
    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, FLAGS_layout, FLAGS_table, FLAGS_det && FLAGS_rec);
#else
    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, args.layout, args.table, args.det && args.rec);
#endif

    for (int j = 0; j < structure_results.size(); j++) {
      std::cout << j << "\ttype: " << structure_results[j].type
                << ", region: [";
      std::cout << structure_results[j].box[0] << ","
                << structure_results[j].box[1] << ","
                << structure_results[j].box[2] << ","
                << structure_results[j].box[3] << "], score: ";
      std::cout << structure_results[j].confidence << ", res: ";

      if (structure_results[j].type == "table") {
        std::cout << structure_results[j].html << std::endl;
#ifdef PPOCR_gflags_ENABLED
        if (structure_results[j].cell_box.size() > 0 && FLAGS_visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
                                   FLAGS_output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
#else
        if (structure_results[j].cell_box.size() > 0 && args.visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
                                   args.output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
#endif
      }  {
        std::cout << "count of ocr result is : "
                  << structure_results[j].text_res.size() << std::endl;
        if (structure_results[j].text_res.size() > 0) {
          std::cout << "********** print ocr result "
                    << "**********" << std::endl;
          Utility::print_result(structure_results[j].text_res);
          std::cout << "********** end print ocr result "
                    << "**********" << std::endl;
        }
      }
    }
  }
#ifdef PPOCR_benchmark_ENABLED
  if (FLAGS_benchmark) {
    engine.benchmark_log(cv_all_img_names.size());
  }
#endif
}

int main(int argc, char **argv) {
  // Parsing command-line
#ifdef PPOCR_gflags_ENABLED
  ppocrParseCommandLineFlags(&argc, &argv, true);
#else
  Args args;
  args.parse(argc, argv);
#endif

  check_params(args);

#ifdef PPOCR_gflags_ENABLED
  if (!Utility::PathExists(FLAGS_image_dir)) {
    std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
              << std::endl;
    exit(1);
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(FLAGS_image_dir, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

  if (!Utility::PathExists(FLAGS_output)) {
    Utility::CreateDir(FLAGS_output);
  }
  if (FLAGS_type == "ocr") {
    ocr(cv_all_img_names);
  } else if (FLAGS_type == "structure") {
    structure(cv_all_img_names);
  } else {
    std::cout << "only value in ['ocr','structure'] is supported" << std::endl;
  }
#else
  if (!Utility::PathExists(args.image_dir)) {
    std::cerr << "[ERROR] image path not exist! image_dir: " << args.image_dir
              << std::endl;
    exit(1);
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(args.image_dir, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

  if (!Utility::PathExists(args.output)) {
    Utility::CreateDir(args.output);
  }
  if (args.type == "ocr") {
    ocr(cv_all_img_names, args);
  } else if (args.type == "structure") {
    structure(cv_all_img_names, args);
  } else {
    std::cout << "only value in ['ocr','structure'] is supported" << std::endl;
  }
#endif
  return 0;
}
#else

int main(int argc, char **argv)
{
  CPPOCR ocr;

  int ret = ppocr_from_args( &ocr, argc, argv );
  if ( ret )
    return ret;

  PPPOcrResult results;
  ret = ppocr_cmd( ocr, &results );
  if ( !ret )
  {
    ppocr_print_result(results);
    ppocr_free( results );
  }

  ppocr_destroy( ocr );

  return ret;
}

#endif
