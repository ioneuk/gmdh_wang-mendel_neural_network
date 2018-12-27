package org.gmdh.neuro.fuzzy.reader;

import org.apache.commons.lang3.StringUtils;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.DataSet;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.utils.NormalizationUtils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class DataReader {

    private static final int ATTRIBUTES = 7;

    public DataSet loadDataSet(int trainSetPercent) {
        List<DataEntry> dataEntries = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/economic.dat")));
            while (true) {
                reader.readLine();
                String line = reader.readLine();
                if (line == null) break;
                DataEntry dataObject = parse(line);
                dataEntries.add(dataObject);
            }
        } catch (Exception e) {
        }

        int trainCount = dataEntries.size() * trainSetPercent / 100;
        int testCount = dataEntries.size() - trainCount;

        NetworkData trainData = new NetworkData();
        NetworkData testData = new NetworkData();
        List<DataEntry> trainEntries = new ArrayList<>(trainCount);
        List<DataEntry> testEntries = new ArrayList<>(testCount);
        trainEntries.addAll(dataEntries.subList(0, trainCount));
        testEntries.addAll(dataEntries.subList(trainCount, trainCount + testCount));
        Collections.shuffle(trainEntries);
        Collections.shuffle(testEntries);
        trainData.setDataEntries(trainEntries);
        testData.setDataEntries(testEntries);

        NormalizationUtils.normalizeDataSet(trainData);
        NormalizationUtils.normalizeDataSet(testData);

        DataSet dataSet = new DataSet();
        dataSet.setTrainData(trainData);
        dataSet.setTestData(testData);
        return dataSet;
    }

    private DataEntry parse(String line) {
        DataEntry dataEntry = new DataEntry();
        String[] tokens = StringUtils.split(line);
        double[] input = new double[ATTRIBUTES];
        for (int i = 0; i < tokens.length - 1; i++) {
            input[i] = Double.parseDouble(tokens[i]);
        }
        dataEntry.setRegressors(input);
        double result = Double.parseDouble(tokens[tokens.length-1]);
        dataEntry.setResult(result);
        return dataEntry;
    }
}

