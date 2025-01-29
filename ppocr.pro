CONFIG -= qt

TEMPLATE = lib

TARGET = ppocr


win32:DEFINES += CPPOCR_LIBRARY

include($$PWD/../nwDeployed/nwDeployed.pri)

OPENCV_DIR = $$PWD/deps/opencv-3.4.20

INCLUDEPATH += . \
    $$PWD/include \
    $${OPENCV_DIR}/include

win32:INCLUDEPATH += $${NWDEP_DIR}/getopt
win32:LIBS += -lgetopt

HEADERS += $$PWD/include/ppocr_c.h

SOURCES += \
    $$PWD/src/ppocr_c.cpp

win32-msvc* {
 LIBS += -L$${OPENCV_DIR}/x64/vc16/lib
}

LIBS += -lpdddle_ocr -lopencv_world3420
