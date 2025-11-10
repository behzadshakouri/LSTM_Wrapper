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
double ComputeMSE(const arma::cube& pred, const arma::cube& Y);
double ComputeR2(const arma::cube& pred, const arma::cube& Y);

/* ============================================================
 *                Normalization
 * ============================================================ */
void ApplyNormalization(NormalizationType mode,
                        arma::mat& train,
                        arma::mat& test,
                        arma::rowvec& mins,
                        arma::rowvec& maxs,
                        bool normalizeOutputs,
                        size_t inputSize,
                        bool normalizeOnlyOutputs);

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
/**
 * @brief SaveResults (unified version)
 *
 * Supports all normalization types (PerVariable, MLpackMinMax, ZScore)
 * and both ASM (IO=false) and IO=true datasets. Saves unscaled input data
 * and corresponding predicted outputs in original physical units.
 */
void SaveResults(const std::string& filename,
                 arma::cube predictions,   // scaled predictions
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 arma::cube X,              // scaled inputs
                 arma::cube Y,              // scaled observed
                 int inputSize,
                 int outputSize,
                 bool IO,
                 bool normalizeOutputs,
                 bool normalizeOnlyOutputs,
                 NormalizationType normType = NormalizationType::PerVariable);

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
