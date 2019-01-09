package org.gmdh.neuro.fuzzy.gmdh.config;

import lombok.Data;

@Data
public class NodeConfig {

    private int regressorsCount;
    private int mFunctionsPerInput;
    private double learningRate;
    private double slope;
}
