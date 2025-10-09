#pragma once
#include <mlpack.hpp>
#include <armadillo>
#include <string>

#include <pch.h>       // ✅ must come first

using namespace mlpack;
using namespace mlpack::data;

// ================= Function Declarations ==================

double ComputeMSE(const arma::cube& pred, const arma::cube& Y);
double ComputeR2(const arma::cube& pred, const arma::cube& Y);

void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X, arma::cube& y,
                          size_t rho, size_t inputSize,
                          size_t outputSize, bool IO);

void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputsize,
                 const int outputsize,
                 const bool IO);
