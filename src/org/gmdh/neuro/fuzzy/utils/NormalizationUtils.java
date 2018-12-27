package org.gmdh.neuro.fuzzy.utils;

import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;

import java.util.List;

public class NormalizationUtils {

    public static void normalizeDataSet(NetworkData networkData) {
        int regressorsCount = networkData.getDataEntries().get(0).getRegressors().length;
        for (int i = 0; i < regressorsCount; i++) {
            normalizeAttribute(networkData.getDataEntries(), i);
        }
    }

    private static void normalizeAttribute(List<DataEntry> dataEntries, int attributeIndex) {
        double max = 0;
        for (DataEntry dataEntry : dataEntries) {
            max = Math.max(max, dataEntry.getRegressors()[attributeIndex]);
        }

        if (max == 0) return;
        for (DataEntry data : dataEntries) {
            data.getRegressors()[attributeIndex] /= max;
        }
    }
}
