// common args
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.")
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.")
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.")
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.")
DEFINE_int32(cpu_threads, 4, "Num of threads with CPU.")
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.")
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8")

// DEFINE_bool(benchmark, false, "Whether use benchmark.")
DEFINE_string(output, "./output", "Save output files log path.")

DEFINE_string(image_dir, "", "Dir of input image.")
DEFINE_int32(type, 1, "Perform function, value in [ 1 - ocr, 2 - structure].")
// detection related
DEFINE_string(det_model_dir, "", "Path of det inference model.")
DEFINE_string(limit_type, "max", "limit_type of input image.")
DEFINE_int32(limit_side_len, 960, "limit_side_len of input image.")
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.")
DEFINE_double(det_db_box_thresh, 0.6, "Threshold of det_db_box_thresh.")
DEFINE_double(det_db_unclip_ratio, 1.5, "Threshold of det_db_unclip_ratio.")
DEFINE_bool(use_dilation, false, "Whether use the dilation on output map.")
DEFINE_string(det_db_score_mode, "slow", "Whether use polygon score.")
DEFINE_bool(visualize, true, "Whether show the detection results.")
// classification related
DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.")
DEFINE_string(cls_model_dir, "", "Path of cls inference model.")
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.")
DEFINE_int32(cls_batch_num, 1, "cls_batch_num.")
// recognition related
DEFINE_string(rec_model_dir, "", "Path of rec inference model.")
DEFINE_int32(rec_batch_num, 6, "rec_batch_num.")
DEFINE_string(rec_char_dict_path, "./ppocr_keys_v1.txt", "Path of dictionary.")
DEFINE_int32(rec_img_h, 48, "rec image height")
DEFINE_int32(rec_img_w, 320, "rec image width")

// layout model related
DEFINE_string(layout_model_dir, "", "Path of table layout inference model.")
DEFINE_string(layout_dict_path, "./dict/layout_dict/layout_publaynet_dict.txt", "Path of dictionary.")
DEFINE_double(layout_score_threshold, 0.5, "Threshold of score.")
DEFINE_double(layout_nms_threshold, 0.5, "Threshold of nms.")
// structure model related
DEFINE_string(table_model_dir, "", "Path of table struture inference model.")
DEFINE_int32(table_max_len, 488, "max len size of input image.")
DEFINE_int32(table_batch_num, 1, "table_batch_num.")
DEFINE_bool(merge_no_span_structure, true, "Whether merge <td> and </td> to <td></td>")
DEFINE_string(table_char_dict_path, "./dict/table_structure_dict_ch.txt", "Path of dictionary.")

// ocr forward related
DEFINE_bool(det, true, "Whether use det in forward.")
DEFINE_bool(rec, true, "Whether use rec in forward.")
DEFINE_bool(cls, false, "Whether use cls in forward.")
DEFINE_bool(table, false, "Whether use table structure in forward.")
DEFINE_bool(layout, false, "Whether use layout analysis in forward.")

DEFINE_void(help, false, "Show this message.")
