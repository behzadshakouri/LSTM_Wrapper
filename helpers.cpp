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
 *                Metric Functions
 * ============================================================ */

double ComputeMSE(const arma::cube& pred, const arma::cube& Y)
{
    return metric::SquaredEuclideanDistance::Evaluate(pred, Y) / Y.n_elem;
}

double ComputeR2(const arma::cube& pred, const arma::cube& Y)
{
    arma::vec yTrue = arma::vectorise(Y);
    arma::vec yPred = arma::vectorise(pred);
    double ssRes = arma::accu(arma::square(yTrue - yPred));
    double ssTot = arma::accu(arma::square(yTrue - arma::mean(yTrue)));
    return 1.0 - ssRes / ssTot;
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

/* ============================================================
 *                Result Saving / Post-Processing
 * ============================================================ */

void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputsize,
                 const int outputsize,
                 const bool IO)
{
    arma::mat flatDataAndPreds = IOData.slice(IOData.n_slices - 1);
    scale.InverseTransform(flatDataAndPreds, flatDataAndPreds);

    arma::mat temp = predictions.slice(predictions.n_slices - 1);
    arma::mat tempExp = temp; // copy for transformation

    // Pad appropriately based on IO mode
    if (!IO)
        tempExp.insert_rows(0, inputsize, 0);   // ASM: add inputs
    else
        tempExp.insert_rows(0, inputsize - outputsize, 0); // IO: add input minus outputs

    scale.InverseTransform(tempExp, tempExp);

    // Ensure consistent row count
    size_t nrows = std::max(flatDataAndPreds.n_rows, tempExp.n_rows);
    if (flatDataAndPreds.n_rows < nrows)
    {
        arma::mat padded = arma::join_cols(flatDataAndPreds,
                                           arma::zeros<arma::mat>(nrows - flatDataAndPreds.n_rows,
                                                                  flatDataAndPreds.n_cols));
        flatDataAndPreds = std::move(padded);
    }

    // Add one empty column for alignment
    tempExp.insert_cols(0, 1, true);
    flatDataAndPreds.insert_cols(flatDataAndPreds.n_cols, 1, true);

    // Append prediction rows safely
    size_t nPredRows = std::min((size_t)outputsize, (size_t)tempExp.n_rows);
    flatDataAndPreds.insert_rows(flatDataAndPreds.n_rows,
                                 tempExp.rows(tempExp.n_rows - nPredRows, tempExp.n_rows - 1));

    data::Save(filename, flatDataAndPreds);

    cout << "Saved predictions to: " << filename << endl;
    cout << "Predicted output (last): ";
    for (int i = 0; i < outputsize; ++i)
        cout << "(" << flatDataAndPreds(flatDataAndPreds.n_rows - outputsize + i,
                                        flatDataAndPreds.n_cols - 1) << ") ";
    cout << endl;
}


