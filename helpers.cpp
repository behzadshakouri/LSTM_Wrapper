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
                 const int inputsize,
                 const int outputsize,
                 const bool IO)
{
    arma::mat flatDataAndPreds = IOData.slice(IOData.n_slices - 1);
    scale.InverseTransform(flatDataAndPreds, flatDataAndPreds);

    arma::mat temp = predictions.slice(predictions.n_slices - 1);
    if (!IO)
        temp.insert_rows(0, inputsize, 0);
    else
        temp.insert_rows(0, inputsize - outputsize, 0);

    scale.InverseTransform(temp, temp);

    temp.insert_cols(0, 1, true);
    flatDataAndPreds.insert_cols(flatDataAndPreds.n_cols, 1, true);

    flatDataAndPreds.insert_rows(
        flatDataAndPreds.n_rows,
        temp.rows(temp.n_rows - outputsize, temp.n_rows - 1)
    );

    mlpack::data::Save(filename, flatDataAndPreds);

    cout << "Saved predictions to: " << filename << endl;
    cout << "The predicted output (last one) is: " << endl;
    for (int i = outputsize - 1; i >= 0; --i)
        cout << " (" << flatDataAndPreds(flatDataAndPreds.n_rows - outputsize + i,
                                         flatDataAndPreds.n_cols - 1) << ") " << endl;
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
