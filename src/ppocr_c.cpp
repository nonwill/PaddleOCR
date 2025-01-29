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
#include "ppocr_c.h"
#include <paddlestructure.h>
#include "opencv2/imgcodecs.hpp"
#include <args.h>

#include <iostream>
#include <vector>

using namespace PaddleOCR;

int check_params( Args const & args ) {
  if (args.det) {
    if (args.det_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      return 1;
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
      return 2;
    }
  }

  if (args.cls && args.use_angle_cls) {
    if (args.cls_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[cls]: ./ppocr "
                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      return 3;
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
      return 4;
    }
  }

  if (args.layout) {
    if (args.layout_model_dir.empty() || args.image_dir.empty()) {
      std::cout << "Usage[layout]: ./ppocr "
                << "--layout_model_dir=/PATH/TO/LAYOUT_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      return 5;
    }
  }

  if (args.precision != "fp32" && args.precision != "fp16" &&
      args.precision != "int8") {
    std::cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    return 6;
  }

  return 0;
}


struct PPOCRC {
  Args args;
  PPOCR *exe;
  bool debug;
};

void ppocr_enable_cout(CPPOCR cppocr, bool yes)
{
  cppocr->debug = yes;
}

int ppocr_from_args( CPPOCR * cppocr, int argc, char ** argv )
{
  Args args;
  args.parse(argc, argv);

  int ret = check_params(args);
  if ( ret )
    return ret;

  PPOCR *ocr;

  if (args.type == "ocr")
    ocr = new PaddleOCR::PPOCR(args);
  else if (args.type == "structure")
  {
    ocr = new PaddleOCR::PaddleStructure(args);
  }
  else
  {
    std::cout << "Only method in ['ocr','structure'] is supported" << std::endl;
    return 7;
  }

  if ( !ocr )
    return -2;

  *cppocr = new struct PPOCRC;
  if ( !(*cppocr) )
  {
    delete ocr;
    return -1;
  }

  (*cppocr)->args = args;
  (*cppocr)->exe = ocr;
  (*cppocr)->debug = false;

  return 0;
}

int ppocr_from_sxml( CPPOCR * cppocr, char const * xmlfile )
{
    return -5;
}


void ppocr_destroy( CPPOCR cppocr )
{
  if ( !cppocr )
    return;

  delete cppocr->exe;
  delete cppocr;
}


int ocr(std::vector<cv::String> &cv_all_img_names, CPPOCR cppocr) {
  Args const & args = cppocr->args;
  PPOCR &ocr = *(cppocr->exe);

  std::vector<cv::Mat> img_list;
  std::vector<cv::String> img_names;
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      if ( cppocr->debug )
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }
    img_list.emplace_back(img);
    img_names.emplace_back(cv_all_img_names[i]);
  }

  if ( img_list.empty() )
    return -8;

  std::vector<std::vector<OCRPredictResult>> ocr_results =
      ocr.ocr(img_list, args.det, args.rec, args.cls);

  if ( !cppocr->debug )
    return 0;

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

  return 0;
}

int structure(std::vector<cv::String> &cv_all_img_names, CPPOCR cppocr) {
  PaddleStructure * _ = dynamic_cast<PaddleStructure*>(cppocr->exe);
  if ( !_ )
    return -3;

  Args const & args = cppocr->args;
  PaddleStructure &engine = *_;

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    if ( cppocr->debug )
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      if ( cppocr->debug )
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }

    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, args.layout, args.table, args.det && args.rec);

    if ( cppocr->debug )
      continue;

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
        if (structure_results[j].cell_box.size() > 0 && args.visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
                                   args.output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
      } else {
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

  return 0;
}


int ppocr_exe( CPPOCR cppocr, char const * image_dir ) {
  Args & args = cppocr->args;
  args.image_dir = image_dir;
  return ppocr_cmd( cppocr );
}

int ppocr_cmd( CPPOCR cppocr ) {
  // Parsing command-line

  if ( !cppocr )
    return -1;
  if ( !cppocr->exe )
    return -2;

  Args const & args = cppocr->args;

  if (!Utility::PathExists(args.image_dir)) {
    if ( cppocr->debug )
    std::cerr << "[ERROR] image path not exist! image_dir: " << args.image_dir
              << std::endl;
    return -3;
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(args.image_dir, cv_all_img_names);
  if ( cppocr->debug )
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

  if (cppocr->debug && !Utility::PathExists(args.output))
    Utility::CreateDir(args.output);

  if (args.type == "ocr")
    return ocr(cv_all_img_names, cppocr);
  else if (args.type == "structure")
    return structure(cv_all_img_names, cppocr);

  if ( cppocr->debug )
    std::cout << "only value in ['ocr','structure'] is supported" << std::endl;

  return -0xffff;
}
