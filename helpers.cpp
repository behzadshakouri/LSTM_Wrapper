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

    if (ss_tot == 0.0)  // prevent division by zero
        return 0.0;

    return 1.0 - (ss_res / ss_tot);
}

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
    int counter = outputsize;
    while (counter > 0)
    {
        cout << " (" << flatDataAndPreds(flatDataAndPreds.n_rows - counter,
                                         flatDataAndPreds.n_cols - 1) << ") " << endl;
        --counter;
    }
}

/* ============================================================
 *                Metric Functions
 * ============================================================ */
/*
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
/*
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
/*
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputsize,
                 const int outputsize,
                 const bool IO)
{
    // Last slice = last timestep for predictions and inputs
    arma::mat flatDataAndPreds = IOData.slice(IOData.n_slices - 1);
    scale.InverseTransform(flatDataAndPreds, flatDataAndPreds);

    arma::mat temp = predictions.slice(predictions.n_slices - 1);

    // --- Match old (working) padding logic ---
    if (!IO)
        temp.insert_rows(0, inputsize, 0.0);             // ASM mode
    else
        temp.insert_rows(0, inputsize - outputsize, 0.0); // IO mode

    scale.InverseTransform(temp, temp);

    // --- Align row counts before merging ---
    const size_t nrows = std::min(flatDataAndPreds.n_rows, temp.n_rows);
    arma::mat merged = flatDataAndPreds.rows(0, nrows - 1);
    arma::mat preds   = temp.rows(0, nrows - 1);

    // Append prediction columns to the right
    merged.insert_cols(merged.n_cols, preds.tail_rows(outputsize).t());

    data::Save(filename, merged);

    cout << "✅ Saved predictions to: " << filename << endl;
    cout << "Last predicted values: ";
    preds.tail_rows(outputsize).t().print();
}
*/
