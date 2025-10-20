#pragma once
#include <string>
#include <cstddef>

#include <pch.h>       // ✅ must come first
#include "helpers.h"   // includes CreateTimeSeriesData, ComputeMSE, ComputeR2, SaveResults

// ============================================================================================
// KFold mode selector (mirrors FFN wrapper)
// ============================================================================================
enum class KFoldMode
{
    Random = 0,      // classic k-fold
    TimeSeries = 1,  // forward-chaining (no lookahead)
    FixedRatio = 2   // fixed training prefix + moving validation fold
};


// ============================================================================================
// Training function declarations
// ============================================================================================

// ---- Single train/test split ----
void TrainSingle(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool ASM,
                 bool bTrain, bool bLoadAndTrain);

// ---- K-Fold (default TimeSeries mode) ----
void TrainKFold(const std::string& dataFile,
                const std::string& modelFile,
                const std::string& predFile_Test,
                const std::string& predFile_Train,
                size_t inputSize, size_t outputSize,
                int rho, int kfolds,
                double stepSize, size_t epochs,
                size_t batchSize, bool IO, bool ASM,
                bool bTrain, bool bLoadAndTrain);

// ---- Extended K-Fold (user-selectable mode + ratio) ----
void TrainKFold_WithMode(const std::string& dataFile,
                         const std::string& modelFile,
                         const std::string& predFile_Test,
                         const std::string& predFile_Train,
                         size_t inputSize, size_t outputSize,
                         int rho, int kfolds,
                         double stepSize, size_t epochs,
                         size_t batchSize, bool IO, bool ASM,
                         bool bTrain, bool bLoadAndTrain,
                         int modeInt,          // 0=Random, 1=TimeSeries, 2=FixedRatio
                         double trainRatio,    // used only for FixedRatio
                         double testHoldout);  // e.g. 0.3 to mirror TrainSingle
