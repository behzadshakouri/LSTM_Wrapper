#ifndef PCH_H
#define PCH_H
#pragma once

// ============================================================
//  Precompiled Header (for mlpack 3.x builds)
// ============================================================

// C++ standard includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Armadillo
#include <armadillo>

// Ensmallen
#include <ensmallen.hpp>

// mlpack (core + ANN + metrics are all included)
#include <mlpack.hpp>

// Qt
#include <QDebug>

// BLAS/LAPACK + SuperLU
#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#define ARMA_USE_SUPERLU

// Enable mlpack ANN serialization
#define MLPACK_ENABLE_ANN_SERIALIZATION

#endif // PCH_H
