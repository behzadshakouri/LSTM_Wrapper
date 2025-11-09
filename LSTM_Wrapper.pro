TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += core

# ---------------- Precompiled Header ----------------
CONFIG += precompile_header
PRECOMPILED_HEADER = pch.h
precompile_header:!isEmpty(PRECOMPILED_HEADER) {
    DEFINES += USING_PCH
}

# ---------------- System Definitions ----------------
DEFINES += GSL
CONFIG  += PowerEdge
DEFINES += PowerEdge

# Uncomment if building on other systems
# CONFIG += Behzad
# DEFINES += Behzad

# CONFIG += Arash
# DEFINES += Arash

# ---------------- Compiler & Linker Flags ----------------
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp

# ---------------- Project Paths ----------------
PowerEdge {
    OHQPATH = /mnt/3rd900/Projects/Utilities
}

Behzad {
    OHQPATH = /home/behzad/Projects/Utilities
}

Arash {
    OHQPATH = /home/arash/Projects/Utilities
}

# ---------------- Include & Library Paths ----------------
INCLUDEPATH += $$OHQPATH
INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/include
INCLUDEPATH += $$HOME/Libraries/ensmallen/include

# ---------------- Libraries ----------------
LIBS += -lmlpack -larmadillo -llapack -lblas -lgsl \
        -lboost_filesystem -lboost_system -lboost_iostreams \
        -lgomp -lpthread

# ---------------- Definitions ----------------
DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS _ARMA
DEFINES += use_VTK ARMA_USE_SUPERLU
CONFIG  += use_VTK

# ---------------- Source Files ----------------
SOURCES += \
    helpers.cpp \
    train_modes.cpp \
    lstmtimeseriesset.cpp \
    main.cpp

# ---------------- Header Files ----------------
HEADERS += \
    helpers.h \
    train_modes.h \
    lstmtimeseriesset.h \
    pch.h
