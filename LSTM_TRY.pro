TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += core

CONFIG+= precompile_header
PRECOMPILED_HEADER = pch.h

precompile_header:!isEmpty(PRECOMPILED_HEADER) {
DEFINES += USING_PCH
}


DEFINES += GSL

CONFIG += Behzad
DEFINES += Behzad

#CONFIG += Arash
#DEFINES += Arash

QMAKE_CXXFLAGS *= "-Xpreprocessor -fopenmp"
QMAKE_LFLAGS +=  -fopenmp


SOURCES += \
    lstmtimeseriesset.cpp \
        main.cpp \


DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS _ARMA




DEFINES += use_VTK ARMA_USE_SUPERLU
CONFIG += use_VTK

INCLUDEPATH += $$OHQPATH
INCLUDEPATH += /usr/local/include

LIBS += -larmadillo -llapack -lblas -lgsl -lboost_filesystem -lboost_system -lboost_iostreams
LIBS += -lgomp -lpthread

HEADERS += \
    lstmtimeseriesset.h \
    pch.h

