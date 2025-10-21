#include "helpers.h"
#include <armadillo>
#include <iostream>
#include <string>

#include <pch.h>       // ✅ must come first
#include "helpers.h"   // now all mlpack headers use the defined macro

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
using namespace ens;
using namespace std;


/* ============================================================
 *                Metric Evaluation
 * ============================================================ */

double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
    return mlpack::metric::SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

double ComputeR2(arma::cube& pred, arma::cube& Y)
{
    arma::vec yTrue = arma::vectorise(Y);
    arma::vec yPred = arma::vectorise(pred);

    double ss_res = arma::accu(arma::square(yTrue - yPred));
    double mean_y = arma::mean(yTrue);
    double ss_tot = arma::accu(arma::square(yTrue - mean_y));

    if (ss_tot == 0.0)
        return 0.0;

    return 1.0 - (ss_res / ss_tot);
}

/* ============================================================
 *                Result Saving / Post-Processing
 * ============================================================ */

void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 mlpack::data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputSize,
                 const int outputSize,
                 const bool IO)
{
    // Get last slice
    arma::mat flatDataAndPreds = IOData.slice(IOData.n_slices - 1);
    arma::mat tempPred = predictions.slice(predictions.n_slices - 1);

    // Try to infer scaler dimension (number of features it was fit on)
    // We assume the scaler was fit on the same shape as its training data
    // So we check last Transform() call argument if available; otherwise fall back to inputSize.
    size_t scalerRows = static_cast<size_t>(inputSize);

    // --- Inverse-transform input features only if possible ---
    if (flatDataAndPreds.n_rows >= scalerRows)
    {
        arma::mat sub = flatDataAndPreds.rows(0, scalerRows - 1);
        scale.InverseTransform(sub, sub);
        flatDataAndPreds.rows(0, scalerRows - 1) = sub;
    }
    else
    {
        scale.InverseTransform(flatDataAndPreds, flatDataAndPreds);
    }

    // --- Prepare prediction block ---
    if (!IO)
        tempPred.insert_rows(0, inputSize, 0.0);
    else
        tempPred.insert_rows(0, inputSize - outputSize, 0.0);

    if (tempPred.n_rows >= scalerRows)
    {
        arma::mat sub = tempPred.rows(0, scalerRows - 1);
        scale.InverseTransform(sub, sub);
        tempPred.rows(0, scalerRows - 1) = sub;
    }
    else
    {
        scale.InverseTransform(tempPred, tempPred);
    }

    // --- Combine and save ---
    tempPred.insert_cols(0, 1, true);
    flatDataAndPreds.insert_cols(flatDataAndPreds.n_cols, 1, true);

    flatDataAndPreds.insert_rows(
        flatDataAndPreds.n_rows,
        tempPred.rows(tempPred.n_rows - outputSize, tempPred.n_rows - 1)
    );

    mlpack::data::Save(filename, flatDataAndPreds);

    // --- Print info ---
    cout << "Saved predictions to: " << filename << endl;
    cout << "The predicted output (last one) is: " << endl;
    for (int i = outputSize - 1; i >= 0; --i)
    {
        cout << " (" << flatDataAndPreds(flatDataAndPreds.n_rows - outputSize + i,
                                         flatDataAndPreds.n_cols - 1) << ") " << endl;
    }
}

/* ============================================================
 *                Time-Series Data Builder
 * ============================================================ */

void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X, arma::cube& y,
                          size_t rho, size_t inputSize,
                          size_t outputSize, bool IO)
{
    if (!IO)
    {
        for (size_t i = 0; i < dataset.n_cols - rho; ++i)
        {
            X.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(0, inputSize - 1),
                               arma::span(i, i + rho - 1));
            y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputSize, inputSize + outputSize - 1),
                               arma::span(i + 1, i + rho));
        }
    }
    else
    {
        for (size_t i = 0; i < dataset.n_cols - rho; ++i)
        {
            X.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(0, inputSize - 1),
                               arma::span(i, i + rho - 1));
            y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputSize - outputSize, inputSize - 1),
                               arma::span(i + 1, i + rho));
        }
    }
}
