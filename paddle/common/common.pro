CONFIG -= qt

TEMPLATE = lib

TARGET = common

DEFINES += PADDLE_NO_PYTHON
win32:DEFINES += PADDLE_WITH_TESTING PADDLE_DLL_EXPORT

include($$PWD/../../../nwDeployed/nwDeployed.pri)

CONFIG -= exceptions

Paddle_ROOT_DIR = $$PWD/..

INCLUDEPATH += . \
    $${Paddle_ROOT_DIR} \
    $${Paddle_ROOT_DIR}/..

HEADERS += $$PWD/*.h

SOURCES += ddim.cc \
    enforce.cc \
    errors.cc \
    flags.cc \
    flags_native.cc \
    performance_statistician.cc
