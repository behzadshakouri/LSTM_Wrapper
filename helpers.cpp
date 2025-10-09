#include "helpers.h"
#include <mlpack.hpp>
#include <armadillo>
#include <iostream>
#include <string>

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
                 mlpack::data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 int inputsize, int outputsize,
                 bool IO)
{
    arma::mat flatData = IOData.slice(IOData.n_slices - 1);
    scale.InverseTransform(flatData, flatData);

    arma::mat pred = predictions.slice(predictions.n_slices - 1);
    pred.insert_rows(0, inputsize, 0);
    scale.InverseTransform(pred, pred);

    // Append columns for alignment with input
    pred.insert_cols(0, 1, true);
    flatData.insert_cols(flatData.n_cols, 1, true);
    flatData.insert_rows(flatData.n_rows,
                         pred.rows(pred.n_rows - outputsize, pred.n_rows - 1));

    data::Save(filename, flatData);
    cout << "Saved predictions to: " << filename << endl;
    cout << "Predicted output (last): ";
    for (int i = outputsize - 1; i >= 0; --i)
        cout << "(" << flatData(flatData.n_rows - 1 - i, flatData.n_cols - 1) << ") ";
    cout << endl;
}
