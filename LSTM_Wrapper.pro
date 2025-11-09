###############################################################
#  LSTM_Wrapper.pro
#  Qt / qmake build configuration for LSTM-based ASM emulator
#  Authors: Behzad Shakouri & Arash Massoudieh
###############################################################

TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += core

# ------------------------------------------------------------
# Precompiled Header
# ------------------------------------------------------------
CONFIG += precompile_header
PRECOMPILED_HEADER = pch.h
precompile_header:!isEmpty(PRECOMPILED_HEADER) {
    DEFINES += USING_PCH
}

# ------------------------------------------------------------
# Build Environment
# ------------------------------------------------------------
# Uncomment your environment
CONFIG += PowerEdge
DEFINES += PowerEdge

# CONFIG += Behzad
# DEFINES += Behzad

# CONFIG += Arash
# DEFINES += Arash

# ------------------------------------------------------------
# Compiler / Linker Flags
# ------------------------------------------------------------
QMAKE_CXXFLAGS *= "-fopenmp"
QMAKE_LFLAGS  += -fopenmp

DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS _ARMA
DEFINES += ARMA_USE_SUPERLU GSL
CONFIG  += use_VTK

# ------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------
INCLUDEPATH += $$OHQPATH

# System include directories
INCLUDEPATH += /usr/include
INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/include/mlpack
INCLUDEPATH += /usr/include/armadillo
INCLUDEPATH += /usr/include/ensmallen
INCLUDEPATH += /usr/include/boost

# (Optional) Local library paths, if you built mlpack manually
# INCLUDEPATH += $$HOME/Libraries/mlpack/include
# INCLUDEPATH += $$HOME/Libraries/ensmallen/include

# ------------------------------------------------------------
# Source Files
# ------------------------------------------------------------
SOURCES += \
    helpers.cpp \
    lstmtimeseriesset.cpp \
    main.cpp \
    train_modes.cpp

# ------------------------------------------------------------
# Header Files
# ------------------------------------------------------------
HEADERS += \
    helpers.h \
    lstmtimeseriesset.h \
    pch.h \
    train_modes.h

# ------------------------------------------------------------
# Libraries (system-installed)
# ------------------------------------------------------------
LIBS += -L/usr/lib/x86_64-linux-gnu \
        -L/usr/local/lib \
        -lmlpack \
        -larmadillo \
        -lboost_serialization \
        -lboost_program_options \
        -lboost_system \
        -lboost_filesystem \
        -lboost_iostreams \
        -llapack -lblas -lgsl \
        -lgomp -lpthread

# ------------------------------------------------------------
# Notes:
#  - Ensure you have installed these packages:
#      sudo apt install libmlpack-dev libarmadillo-dev \
#           libensmallen-dev libboost-all-dev libgsl-dev
#  - No mlpack source files are compiled each time — only linked.
# ------------------------------------------------------------
