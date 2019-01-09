package org.gmdh.neuro.fuzzy.gmdh.data;

import lombok.Data;

@Data
public class DataSet {

    private NetworkData trainData;
    private NetworkData testData;

    public double getMaxOutput() {
        return Math.max(trainData.getMaxOutput(), testData.getMaxOutput());
    }
}
