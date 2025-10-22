#include "helpers.h"
#include <armadillo>
#include <iostream>
#include <string>

#include <pch.h>       // must come first
#include "helpers.h"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
using namespace ens;
using namespace std;

/* ============================================================
 *                Shape Validator (Debug Utility)
 * ============================================================ */
void ValidateShapes(const arma::mat& dataset,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    size_t rho)
{
    cout << "\n========== [ValidateShapes Debug Info] ==========\n";
    cout << "Dataset: " << dataset.n_rows << "x" << dataset.n_cols << endl;
    cout << "X cube : " << X.n_rows << "x" << X.n_cols << "x" << X.n_slices << endl;
    cout << "Y cube : " << Y.n_rows << "x" << Y.n_cols << "x" << Y.n_slices << endl;
    cout << "inputSize=" << inputSize << "  outputSize=" << outputSize
         << "  rho=" << rho << endl;

    if (dataset.n_rows < inputSize)
        cerr << "⚠️  Dataset rows (" << dataset.n_rows
             << ") < inputSize (" << inputSize << "). Check your data layout.\n";

    if (Y.n_rows != outputSize)
        cerr << "⚠️  Y cube row mismatch: expected " << outputSize
             << ", got " << Y.n_rows << endl;

    if (X.n_cols != Y.n_cols)
        cerr << "⚠️  Column mismatch between X and Y cubes: "
             << X.n_cols << " vs " << Y.n_cols << endl;

    if (X.n_slices != Y.n_slices)
        cerr << "⚠️  Slice mismatch between X and Y cubes: "
             << X.n_slices << " vs " << Y.n_slices << endl;

    cout << "==============================================\n";
}

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
 *                Safe SaveResults() with Debug Logging
 * ============================================================ */
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 mlpack::data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputSize,
                 const int outputSize,
                 const bool IO)
{
    cout << "\n========== [SaveResults Debug Info] ==========\n";

    // Extract last slices
    arma::mat xSlice = IOData.slice(IOData.n_slices - 1);           // features (e.g. 9×N)
    arma::mat yPred  = predictions.slice(predictions.n_slices - 1); // outputs (e.g. 1×N)

    cout << "xSlice size: " << xSlice.n_rows << "x" << xSlice.n_cols << endl;
    cout << "yPred  size: " << yPred.n_rows  << "x" << yPred.n_cols  << endl;

    const arma::uword N = xSlice.n_cols;

    // --- Build combined block matching the scaler's Fit() layout ---
    arma::mat combined;
    if (!IO)
    {
        // Training likely used full dataset [inputs; outputs]
        combined.set_size(inputSize + outputSize, N);
        combined.rows(0, inputSize - 1) = xSlice;
        combined.rows(inputSize, inputSize + outputSize - 1) = yPred;
    }
    else
    {
        // IO layout: outputs occupy last outputSize rows of input block
        combined.set_size(inputSize, N);
        combined.rows(0, inputSize - outputSize - 1) = xSlice.rows(0, inputSize - outputSize - 1);
        combined.rows(inputSize - outputSize, inputSize - 1) = yPred;
    }

    cout << "Combined pre-inverse size: " << combined.n_rows << "x" << combined.n_cols << endl;

    // --- Inverse-transform full combined block safely ---
    bool inverseOk = true;
    try {
        scale.InverseTransform(combined, combined);
    } catch (const std::exception& e) {
        inverseOk = false;
        cout << "⚠️  InverseTransform(combined) failed: " << e.what()
             << "  -> Saving scaled values instead.\n";
    }

    // --- Split back inputs and predictions ---
    arma::mat X_inv, Yhat_inv;
    if (!IO)
    {
        X_inv    = combined.rows(0, inputSize - 1);
        Yhat_inv = combined.rows(inputSize, inputSize + outputSize - 1);
    }
    else
    {
        X_inv    = combined.rows(0, inputSize - 1);
        Yhat_inv = combined.rows(inputSize - outputSize, inputSize - 1);
    }

    // --- Pad predictions for CSV layout AFTER inverse-transform ---
    arma::mat predBlock = Yhat_inv;  // outputSize×N
    if (!IO)
        predBlock.insert_rows(0, inputSize, 0.0);       // Non-IO: prepend full input block
    else
        predBlock.insert_rows(0, inputSize - outputSize, 0.0);

    // --- Merge blocks and save ---
    predBlock.insert_cols(0, 1, true);                  // blank leading column
    arma::mat table = X_inv;                            // start with inputs
    table.insert_cols(table.n_cols, 1, true);           // blank trailing column

    // Append predicted outputs as bottom rows
    table.insert_rows(
        table.n_rows,
        predBlock.rows(predBlock.n_rows - outputSize, predBlock.n_rows - 1)
    );

    // Align columns if mismatch
    if (table.n_cols != predBlock.n_cols)
    {
        arma::uword minCols = std::min(table.n_cols, predBlock.n_cols);
        cout << "⚠️  Column mismatch. Truncating to " << minCols << " columns.\n";
        table     = table.cols(0, minCols - 1);
        predBlock = predBlock.cols(0, minCols - 1);
    }

    // --- Save to CSV ---
    if (!mlpack::data::Save(filename, table))
    {
        cerr << "❌ Error: Failed to save results to " << filename << endl;
        return;
    }

    cout << "✅ Saved predictions to: " << filename << endl;
    cout << "Predicted output values (last " << outputSize << " rows):\n";
    for (int i = outputSize - 1; i >= 0; --i)
    {
        cout << " (" << table(table.n_rows - outputSize + i,
                               table.n_cols - 1) << ") " << endl;
    }
    cout << "==============================================\n";
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
