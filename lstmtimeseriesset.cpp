#include "lstmtimeseriesset.h"

LSTMTimeSeriesSet::LSTMTimeSeriesSet() : arma::mat() {}

// Copy constructor
LSTMTimeSeriesSet::LSTMTimeSeriesSet(const LSTMTimeSeriesSet& other) : arma::mat(other) {}

// Assignment operator
LSTMTimeSeriesSet& LSTMTimeSeriesSet::operator=(const LSTMTimeSeriesSet& other)
{
    if (this != &other) {
        // Only assign if it's not self-assignment
        arma::mat::operator=(other);
    }
    return *this;
}

LSTMTimeSeriesSet::LSTMTimeSeriesSet(const arma::mat& other) : arma::mat(other) {}

// Assignment operator
LSTMTimeSeriesSet& LSTMTimeSeriesSet::operator=(const arma::mat& other)
{
    if (this != &other) {
        // Only assign if it's not self-assignment
        arma::mat::operator=(other);
    }
    return *this;
}

// Destructor
LSTMTimeSeriesSet::~LSTMTimeSeriesSet() {
    // No need to explicitly free memory since arma::mat handles that.
    // If there were other dynamically allocated resources in the class, you would clean them here.
}
