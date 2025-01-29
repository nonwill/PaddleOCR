
CONFIG -= qt

TEMPLATE = lib

TARGET = pdddle_ocr


mac:CONFIG += static

win32:DEFINES += PPOCR_LIBRARY
#win32:DEFINES += PPOCR_benchmark_ENABLED
#win32:DEFINES += PPOCR_gflags_ENABLED
#CONFIG += Clipper2Lib

include($$PWD/../nwDeployed/nwDeployed.pri)

CONFIG += ppmkl

# https://zhuanlan.zhihu.com/p/680229436

OPENCV_DIR = $$PWD/deps/opencv-3.4.20
PADDLE_LIB = $$PWD/deps/paddle_inference-2.6.2
#PADDLE_LIB = $$PWD/paddle_inference-3.0.0rc0

INCLUDEPATH += . \
    $$PWD/include \
    $${OPENCV_DIR}/include \
    $${PADDLE_LIB}/paddle/include \
    $${NWDEP_DIR}/getopt

Clipper2Lib {
    DEFINES += ClipperLib_Version_2
    msvc*:QMAKE_CXXFLAGS = -std:c++17
    INCLUDEPATH += $${NWDEP_DIR}/Clipper2/include
    LIBS += -lClipper2
} else {
    SOURCES += $$PWD/src/clipper.cpp
    HEADERS += $$PWD/include/clipper.h \
}

# PaddleOCR-2.9.1

SOURCES += \
    $$PWD/src/args.cpp \
    $$PWD/src/ocr_cls.cpp \
    $$PWD/src/ocr_det.cpp \
    $$PWD/src/ocr_rec.cpp \
    $$PWD/src/paddleocr.cpp \
    $$PWD/src/paddlestructure.cpp \
    $$PWD/src/postprocess_op.cpp \
    $$PWD/src/preprocess_op.cpp \
    $$PWD/src/structure_layout.cpp \
    $$PWD/src/structure_table.cpp \
    $$PWD/src/utility.cpp

HEADERS += $$PWD/include/ppocr_api.h \
    $$PWD/include/args.h \
    $$PWD/include/ocr_cls.h \
    $$PWD/include/ocr_det.h \
    $$PWD/include/ocr_rec.h \
    $$PWD/include/paddleocr.h \
    $$PWD/include/paddlestructure.h \
    $$PWD/include/postprocess_op.h \
    $$PWD/include/preprocess_op.h \
    $$PWD/include/structure_layout.h \
    $$PWD/include/structure_table.h \
    $$PWD/include/utility.h

win32:equals(ARCHITECTURE,x64) {
  msvc* {
    LIBS += -L$${OPENCV_DIR}/x64/vc16/lib
    _ = $$nwCopyFileToDir($${OPENCV_DIR}/x64/vc16/bin/*.dll,$${DESTDIR})
  } else {
    LIBS += -L$${OPENCV_DIR}/x64/mingw/lib
    _ = $$nwCopyFileToDir($${OPENCV_DIR}/x64/mingw/bin/*.dll,$${DESTDIR})
  }

  ppmkl {
    _ = $$nwCopyFileToDir($$PWD/deps/oneDNN-3.6.2/bin/dnnl.dll,$${DESTDIR},mkldnn.dll)
    _ = $$nwCopyFileToDir($$PWD/deps/mklml/lib/win-x64/native/*.dll,$${DESTDIR})
    _ = $$nwCopyFileToDir($${PADDLE_LIB}/deps/paddle/lib/mkl/*.dll,$${DESTDIR})
    _ = $$nwCopyFileToDir($${PADDLE_LIB}/deps/paddle/lib/mkl/*.manifest,$${DESTDIR})
    LIBS += -L$${PADDLE_LIB}/paddle/lib/mkl
  } else {
    _ = $$nwCopyFileToDir($$PWD/deps/OpenBLAS-0.3.29/bin/openblas.dll,$${DESTDIR})
    _ = $$nwCopyFileToDir($${PADDLE_LIB}/paddle/lib/*.dll,$${DESTDIR})
    _ = $$nwCopyFileToDir($${PADDLE_LIB}/paddle/lib/*.manifest,$${DESTDIR})
    LIBS += -L$${PADDLE_LIB}/paddle/lib
  }
}

LIBS += -lpaddle_inference -lopencv_world3420

win32:LIBS += -lgetopt

