#pragma once
#include <mlpack.hpp>
#include <armadillo>
#include <string>

#include <pch.h>       // ✅ must come first

using namespace mlpack;
using namespace mlpack::data;

// ===== Metrics =====
double ComputeMSE(arma::cube& pred, arma::cube& Y);
double ComputeR2(arma::cube& pred, arma::cube& Y);

// ===== Data prep =====
template<typename InputDataType = arma::mat,
         typename DataType = arma::cube,
         typename LabelType = arma::cube>
void CreateTimeSeriesData(InputDataType dataset,
                          DataType& X,
                          LabelType& y,
                          const size_t rho,
                          const int inputsize,
                          const int outputsize,
                          const bool IO)
{
    if (!IO)
    {
        for (size_t i = 0; i < dataset.n_cols - rho; ++i)
        {
            X.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(0, inputsize - 1), arma::span(i, i + rho - 1));
            y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputsize, inputsize + outputsize - 1),
                               arma::span(i + 1, i + rho));
        }
    }
    else
    {
        for (size_t i = 0; i < dataset.n_cols - rho; ++i)
        {
            X.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(0, inputsize - 1), arma::span(i, i + rho - 1));
            y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputsize - outputsize, inputsize - 1),
                               arma::span(i + 1, i + rho));
        }
    }
}

// ===== Save results =====
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 mlpack::data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputsize,
                 const int outputsize,
                 const bool IO);

/*
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
*/
