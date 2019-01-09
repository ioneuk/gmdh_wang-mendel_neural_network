package org.gmdh.neuro.fuzzy.utils;

import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;

import java.util.List;


public class NormalizationUtils {

    public static void applyActivationFunction(NetworkData networkData, double slope) {
        for (DataEntry dataEntry : networkData.getDataEntries()) {
            double[] regressors = dataEntry.getRegressors();
            for (int i = 0; i < regressors.length; ++i) {
                regressors[i] = sigmoid(regressors[i], slope);
            }
        }
    }

    public static double[] applyActivationFunction(double[] inputs, double slope) {
        double[] result = new double[inputs.length];
        for(int i = 0; i < inputs.length; ++i) {
            result[i] = sigmoid(inputs[i], slope);
        }
        return result;
    }

    public static void normalizeDataSet(NetworkData networkData) {
        for (int i = 0; i < networkData.getRegressorsCount(); i++) {
            normalizeAttribute(networkData.getDataEntries(), i);
        }
        //normalizeOutput(networkData.getDataEntries());
    }

    public static double sigmoid(double x, double slope) {
        return (1.0 / (1 + Math.exp(-slope * x)));
    }

    private static void normalizeAttribute(List<DataEntry> dataEntries, int attributeIndex) {
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for (DataEntry dataEntry : dataEntries) {
            double attribute = dataEntry.getRegressors()[attributeIndex];
            max = Math.max(max, attribute);
            min = Math.min(min, attribute);
        }

        if (max == 0) return;
        for (DataEntry data : dataEntries) {
            data.getRegressors()[attributeIndex] = (data.getRegressors()[attributeIndex] - min) / (max - min);
        }
    }

    private static void normalizeOutput(List<DataEntry> dataEntries) {
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for (DataEntry dataEntry : dataEntries) {
            double outputValue = dataEntry.getResult();
            max = Math.max(Math.abs(max), Math.abs(outputValue));
            min = Math.min(Math.abs(min), Math.abs(outputValue));
        }

        if (max == 0) return;
        for (DataEntry data : dataEntries) {
            double normalizedOutput = (data.getResult() - min) / (max - min);
            data.setResult(normalizedOutput);
        }
    }
}
