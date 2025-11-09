/**
 * @file helpers.h
 * @brief Utility functions for LSTM Wrapper (normalization, metrics, IO, etc.)
 *
 * Authors:
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#pragma once
#include <armadillo>
#include <string>
#include <iostream>

// ============================================================
//  Normalization type definition
// ============================================================
enum class NormalizationType
{
    PerVariable = 0,   ///< Custom per-row min–max scaling
    MLpackMinMax = 1,  ///< mlpack’s built-in MinMaxScaler
    ZScore       = 2,  ///< Standardization (mean/std)
    None         = 3   ///< No normalization
};

// ============================================================
//  Function declarations
// ============================================================

// Metric utilities
double ComputeMSE(arma::cube& pred, arma::cube& Y);
double ComputeR2(arma::cube& pred, arma::cube& Y);

// Normalization
void ApplyNormalization(NormalizationType mode,
                        arma::mat& train,
                        arma::mat& test,
                        arma::rowvec& mins,
                        arma::rowvec& maxs,
                        bool normalizeOutputs,
                        size_t inputSize);

// Shape and dataset utilities
void ValidateShapes(const arma::mat& data,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    int rho);

void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X,
                          arma::cube& Y,
                          int rho,
                          int inputSize,
                          int outputSize,
                          bool IO);

void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 const arma::cube& X,
                 int inputSize,
                 int outputSize,
                 bool IO,
                 bool normalizeOutputs);

void PrintRunConfig(bool ASM, bool IO, bool bTrain, bool bLoadAndTrain,
                    size_t inputSize, size_t outputSize, int rho,
                    double stepSize, size_t epochs, size_t batchSize,
                    int H1, int H2, int H3,
                    int mode, int kfoldMode, int KFOLDS,
                    double trainRatio, double testHoldout,
                    const std::string& dataFile,
                    const std::string& modelFile,
                    const std::string& predFile_Test,
                    const std::string& predFile_Train);

void ValidateConfigOrWarn(int mode, int kfoldMode, int KFOLDS,
                          double trainRatio, double testHoldout);
