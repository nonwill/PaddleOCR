CONFIG -= qt
TEMPLATE = app

TARGET = ppocr

CONFIG += console

include($$PWD/../nwDeployed/nwDeployed.pri)

CONFIG += ppocr_capi

INCLUDEPATH += . \
    $$PWD/include

SOURCES += $$PWD/src/main.cpp

ppocr_capi {
  LIBS += -lppocr
} else {
  DEFINES += USING_PPOCR_CPP_API
  OPENCV_DIR = $$PWD/deps/opencv-3.4.20
  INCLUDEPATH += $${OPENCV_DIR}/include
  win32-msvc* {
   LIBS += -L$${OPENCV_DIR}/x64/vc16/lib
  }
  LIBS += -lpdddle_ocr -lopencv_world3420
}


QMAKE_POST_LINK = $${DESTDIR}/ppocr.exe \
    --enable_mkldnn=1 --use_dilation=0 \
    --det_model_dir=$$PWD/deps/PP-Modal/ch_PP-OCRv4_det_infer \
    --rec_model_dir=$$PWD/deps/PP-Modal/ch_PP-OCRv4_rec_infer \
    --cls_model_dir=$$PWD/deps/PP-Modal/ch_ppocr_mobile_v2.0_cls_infer \
    --rec_char_dict_path=$$PWD/deps/PP-Modal/ppocr_keys_v1.txt \
    --image_dir=$$PWD/tests --output=$${DESTDIR}/output-use_dilation_0 & \
    $${DESTDIR}/ppocr.exe \
    --enable_mkldnn=1 --use_dilation=1 \
    --det_model_dir=$$PWD/deps/PP-Modal/ch_PP-OCRv4_det_infer \
    --rec_model_dir=$$PWD/deps/PP-Modal/ch_PP-OCRv4_rec_infer \
    --cls_model_dir=$$PWD/deps/PP-Modal/ch_ppocr_mobile_v2.0_cls_infer \
    --rec_char_dict_path=$$PWD/deps/PP-Modal/ppocr_keys_v1.txt \
    --image_dir=$$PWD/tests --output=$${DESTDIR}/output-use_dilation_1

# QMAKE_POST_LINK = $${DESTDIR}/ppocr.exe \
#     --det_model_dir=$$PWD/PP-Modal/ch_PP-OCRv3_det_infer \
#     --rec_model_dir=$$PWD/PP-Modal/ch_PP-OCRv3_rec_infer \
#     --cls_model_dir=$$PWD/PP-Modal/ch_ppocr_mobile_v2.0_cls_infer \
#     --rec_char_dict_path=$${DESTDIR}/ppocr_keys_v1.txt \
#     --image_dir=$$PWD/test.jpg
