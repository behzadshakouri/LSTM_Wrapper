#ifndef LSTMTIMESERIESSET_H
#define LSTMTIMESERIESSET_H

#include "armadillo"

class LSTMTimeSeriesSet : public arma::mat
{
public:
    LSTMTimeSeriesSet();

    // Copy constructor
    LSTMTimeSeriesSet(const LSTMTimeSeriesSet& other);

    LSTMTimeSeriesSet(const arma::mat& other);

    // Assignment operator
    LSTMTimeSeriesSet& operator=(const LSTMTimeSeriesSet& other);

    LSTMTimeSeriesSet& operator=(const arma::mat& other);

    // Destructor
    ~LSTMTimeSeriesSet();
};

#endif // LSTMTIMESERIESSET_H
