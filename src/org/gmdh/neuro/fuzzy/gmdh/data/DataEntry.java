package org.gmdh.neuro.fuzzy.gmdh.data;

import lombok.Data;

import java.util.Arrays;

@Data
public class DataEntry {

    double[] regressors;
    double result;

    public double getInputByColumnNumber(int columnNumber) {
        return regressors[columnNumber];
    }

    @Override
    public DataEntry clone() {
        DataEntry cloneDataEntry = new DataEntry();
        cloneDataEntry.setResult(this.result);
        cloneDataEntry.setRegressors(this.regressors);
        return cloneDataEntry;
    }

    public double getMaxInput() {
        return Arrays.stream(regressors).max().getAsDouble();
    }
}
