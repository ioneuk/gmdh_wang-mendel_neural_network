package org.gmdh.neuro.fuzzy.gmdh.data;

import lombok.Data;

@Data
public class DataEntry {

    double[] regressors;
    double result;

    public double getInputByColumnNumber(int columnNumber) {
        return regressors[columnNumber];
    }
}
