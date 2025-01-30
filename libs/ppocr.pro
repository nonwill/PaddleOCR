CONFIG -= qt

TEMPLATE = lib

TARGET = ppocr

win32:DEFINES += CPPOCR_LIBRARY

include($$PWD/../../nwDeployed/nwDeployed.pri)

PPOCR_ROOT_DIR = $$PWD/..

OPENCV_DIR = $${PPOCR_ROOT_DIR}/deps/opencv-3.4.20

INCLUDEPATH += . \
    $${PPOCR_ROOT_DIR}/include \
    $${OPENCV_DIR}/include

HEADERS += $${PPOCR_ROOT_DIR}/include/ppocr_c.h

SOURCES += \
    $${PPOCR_ROOT_DIR}/src/ppocr_c.cpp

win32-msvc* {
    LIBS += -L$${OPENCV_DIR}/x64/vc16/lib
}

LIBS += -lpdddleocr -lopencv_world3420
