package org.gmdh.neuro.fuzzy.gmdh.config;

import lombok.Data;

@Data
public class GmdhConfig {

    private int regressorsCount;
    private int mFunctionsPerInput;

    private double learnTestRatio;

    private int maxNeuronCountPerLayer;
    private double learningRate;
    private double slope;
}
