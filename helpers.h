/**
 * @file helpers.h
 * @brief Utility function declarations for LSTM Wrapper
 *        (normalization, metrics, time-series creation, and logging).
 */

#pragma once

#include <armadillo>
#include <QString>
#include <QDebug>
#include <string>

/* ============================================================
 *                Normalization Types
 * ============================================================ */
enum class NormalizationType
{
    PerVariable = 0,
    MLpackMinMax = 1,
    ZScore = 2,
    None = 3
};

/* ============================================================
 *                Metrics
 * ============================================================ */
double ComputeMSE(arma::cube& pred, arma::cube& Y);
double ComputeR2(arma::cube& pred, arma::cube& Y);

/* ============================================================
 *                Normalization
 * ============================================================ */
void ApplyNormalization(NormalizationType mode,
                        arma::mat& train,
                        arma::mat& test,
                        arma::rowvec& mins,
                        arma::rowvec& maxs,
                        bool normalizeOutputs,
                        size_t inputSize);

/* ============================================================
 *                Time-Series Cube Preparation
 * ============================================================ */
void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X,
                          arma::cube& Y,
                          int rho,
                          int inputSize,
                          int outputSize,
                          bool IO);

/* ============================================================
 *                Validation and Logging
 * ============================================================ */
void ValidateShapes(const arma::mat& data,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    int rho);

/* ============================================================
 *                Output Handling
 * ============================================================ */
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 const arma::cube& X,
                 int inputSize,
                 int outputSize,
                 bool IO,
                 bool normalizeOutputs);

/* ============================================================
 *                Run Configuration Logging
 * ============================================================ */
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
