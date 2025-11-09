# ============================================================
#  Project: LSTM_Wrapper
#  Description: LSTM surrogate model training (mlpack + Armadillo + Qt)
# ============================================================

TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += core

# ------------------------
# Precompiled header
# ------------------------
CONFIG += precompile_header
PRECOMPILED_HEADER = pch.h

DEFINES += USING_PCH

# ------------------------
# Build configurations
# ------------------------
CONFIG += PowerEdge
DEFINES += PowerEdge

# Uncomment as needed for other systems
# CONFIG += Behzad
# DEFINES += Behzad
# CONFIG += Arash
# DEFINES += Arash

# ------------------------
# Compiler / Linker flags
# ------------------------
QMAKE_CXXFLAGS += -O2 -fopenmp
QMAKE_LFLAGS   += -fopenmp

DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS ARMA_USE_SUPERLU _ARMA
DEFINES += GSL
CONFIG  += use_VTK

# ------------------------
# Include paths
# ------------------------
INCLUDEPATH += /usr/include \
                /usr/local/include \
                $$HOME/Libraries/ensmallen/include

# ------------------------
# Library paths and link order
# ------------------------
LIBS += -L/lib/x86_64-linux-gnu \
        -lmlpack \
        -larmadillo \
        -lboost_system \
        -lboost_filesystem \
        -lboost_serialization \
        -lboost_program_options \
        -lblas -llapack \
        -lgsl -lgslcblas \
        -lgomp -lpthread

# ------------------------
# Source files
# ------------------------
SOURCES += \
    helpers.cpp \
    lstmtimeseriesset.cpp \
    train_modes.cpp \
    main.cpp

# ------------------------
# Header files
# ------------------------
HEADERS += \
    helpers.h \
    lstmtimeseriesset.h \
    pch.h \
    train_modes.h

# ------------------------
# Additional build info
# ------------------------
QMAKE_POST_LINK += echo "✅ Build completed successfully."
