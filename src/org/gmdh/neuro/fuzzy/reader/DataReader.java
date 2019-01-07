package org.gmdh.neuro.fuzzy.reader;

import org.apache.commons.lang3.StringUtils;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.openxml4j.opc.OPCPackage;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.DataSet;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.utils.NormalizationUtils;

import java.io.*;
import java.util.*;

public class DataReader {

    private HSSFWorkbook workbook;

    public DataSet loadDefaultDataSet(int trainSetPercent, int attributesCount) {
        List<DataEntry> dataEntries = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/economic.dat")));
            while (true) {
                String line = reader.readLine();
                if (line == null) break;
                DataEntry dataObject = parse(line, attributesCount);
                dataEntries.add(dataObject);
            }
        } catch (Exception e) {
        }

        return splitDataSetOnTrainTest(trainSetPercent, dataEntries);
    }

    public DataSet loadDataSet(File file, int trainSetPercent, int attributesCount) {
        List<DataEntry> dataEntries = new ArrayList<>();
        try {
            workbook = new HSSFWorkbook(new FileInputStream(file));
            HSSFSheet sheet = workbook.getSheetAt(0);
            int counter = 0;

            while (counter < sheet.getPhysicalNumberOfRows()) {
                double[] inputs = new double[attributesCount];
                DataEntry dataEntry = new DataEntry();
                for (int i = 0; i < attributesCount; ++i) {
                    HSSFRow line = sheet.getRow(counter);
                    inputs[i] = Double.parseDouble(line.getCell(0).toString());
                    ++counter;
                }
                dataEntry.setRegressors(inputs);
                String result = sheet.getRow(counter).getCell(0).toString();
                ++counter;
                dataEntry.setResult(Double.parseDouble(result));
                dataEntries.add(dataEntry);
            }
        } catch (Exception ex) {}
        return splitDataSetOnTrainTest(trainSetPercent, dataEntries);
    }

    private DataSet splitDataSetOnTrainTest(int trainSetPercent, List<DataEntry> dataEntries) {
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

    private DataEntry parse(String line, int attributesCount) {
        DataEntry dataEntry = new DataEntry();
        String[] tokens = StringUtils.split(line);
        double[] input = new double[attributesCount];
        for (int i = 0; i < tokens.length - 1; i++) {
            input[i] = Double.parseDouble(tokens[i]);
        }
        dataEntry.setRegressors(input);
        double result = Double.parseDouble(tokens[tokens.length-1]);
        dataEntry.setResult(result);
        return dataEntry;
    }
}

