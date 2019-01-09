package org.gmdh.neuro.fuzzy;

import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import lombok.Data;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.gmdh.neuro.fuzzy.gmdh.GmdhNeuroFuzzyNetwork;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.DataSet;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.reader.DataReader;
import org.gmdh.neuro.fuzzy.utils.MathUtils;
import org.gmdh.neuro.fuzzy.utils.NormalizationUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;

public class Controller {

    @FXML
    private Slider learnTestRatio;
    @FXML
    private Spinner<Integer> mFunctionsPerInput;
    @FXML
    private Spinner<Double> learningRate;
    @FXML
    private Spinner<Integer> bestNeuronCount;
    @FXML
    private LineChart lineChart;
    @FXML
    private LineChart lineChart2;
    @FXML
    private Spinner<Integer> inputNumber;

    @FXML
    private TableView tableView;
    @FXML
    private TableColumn pointTab;
    @FXML
    private TableColumn realValueTab;
    @FXML
    private TableColumn predictedValueTab;
    @FXML
    private TableColumn deltaTab;
    @FXML
    private TableColumn squaredDeltaTab;
    @FXML
    private TableColumn mapeDeltaTab;
    @FXML
    private TableColumn rmseDeltaTab;
    @FXML
    private Button xlsExport;

    private FileChooser fileChooser;

    private GmdhConfig gmdhConfig;
    private DataReader dataReader;
    private DataSet currentDataSet;
    private GmdhNeuroFuzzyNetwork network;
    private DecimalFormat decimalFormat;

    @FXML
    public void initialize() {
        SpinnerValueFactory.IntegerSpinnerValueFactory mFunctionsValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(0, 10, 6);
        mFunctionsPerInput.setValueFactory(mFunctionsValueFactory);
        SpinnerValueFactory.DoubleSpinnerValueFactory learningRateValueFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 3.0, 2.25, 0.25);
        learningRate.setValueFactory(learningRateValueFactory);
        SpinnerValueFactory.IntegerSpinnerValueFactory bestNeuronCountValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 1000, 20);
        bestNeuronCount.setValueFactory(bestNeuronCountValueFactory);
        SpinnerValueFactory.IntegerSpinnerValueFactory inputNumberValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 20, 4);
        inputNumber.setValueFactory(inputNumberValueFactory);

        pointTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Integer>("point"));
        realValueTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("realValue"));
        predictedValueTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("predictedValue"));
        deltaTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("delta"));
        squaredDeltaTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("squaredDelta"));
        mapeDeltaTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("mapeDelta"));
        rmseDeltaTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("rmseDelta"));

        dataReader = new DataReader();
        fileChooser = new FileChooser();
        FileChooser.ExtensionFilter extFilter =
                new FileChooser.ExtensionFilter("Excel files (*.xls)", "*.xls", "*.xlsx");
        fileChooser.getExtensionFilters().add(extFilter);
        tableView.setEditable(true);
        lineChart.setAnimated(false);
        lineChart2.setAnimated(false);
    }

    @FXML
    public void onTrainAction() {
        GmdhConfig gmdhConfig = new GmdhConfig();
        gmdhConfig.setMaxNeuronCountPerLayer(bestNeuronCount.getValue());
        gmdhConfig.setLearningRate(learningRate.getValue());
        gmdhConfig.setLearnTestRatio(learnTestRatio.getValue());
        gmdhConfig.setRegressorsCount(2);
        gmdhConfig.setMFunctionsPerInput(mFunctionsPerInput.getValue());
        gmdhConfig.setSlope(1 / currentDataSet.getMaxOutput());
        network = new GmdhNeuroFuzzyNetwork(gmdhConfig);

        NetworkData trainData = currentDataSet.getTrainData();
        NetworkData testData = currentDataSet.getTestData();
        NormalizationUtils.normalizeDataSet(trainData);
        NormalizationUtils.normalizeDataSet(testData);
        network.buildGmdhStructure(currentDataSet.getTrainData(), testData);

        drawMseChart(testData);
        drawComparisonChart(testData);
        fillTable(testData);
        enableExportButton();
    }

    private void enableExportButton() {
        xlsExport.setVisible(true);
        xlsExport.setDisable(false);
    }

    @FXML
    public void onLoadDataAction() {
        File file = fileChooser.showOpenDialog(new Stage());
        if (file != null) {
            currentDataSet = dataReader.loadDataSet(file, (int) learnTestRatio.getValue(), inputNumber.getValue());
        } else {
            currentDataSet = dataReader.loadDefaultDataSet((int) learnTestRatio.getValue(), inputNumber.getValue());
        }
    }

    @FXML
    public void onExportAction() throws IOException {
        HSSFWorkbook workbook = new HSSFWorkbook();
        HSSFSheet spreadsheet = workbook.createSheet("report");

        HSSFRow row = spreadsheet.createRow(0);

        for (int j = 0; j < tableView.getColumns().size(); j++) {
            row.createCell(j).setCellValue(((TableColumn)tableView.getColumns().get(j)).getText());
        }

        for (int i = 0; i < tableView.getItems().size(); i++) {
            row = spreadsheet.createRow(i + 1);
            for (int j = 0; j < tableView.getColumns().size(); j++) {
                if(((TableColumn)tableView.getColumns().get(j)).getCellData(i) != null) {
                    row.createCell(j).setCellValue(((TableColumn)tableView.getColumns().get(j)).getCellData(i).toString());
                }
                else {
                    row.createCell(j).setCellValue("");
                }
            }
        }

        File file = fileChooser.showSaveDialog(new Stage());
        if (file != null) {
            FileOutputStream fileOut = new FileOutputStream(file);
            workbook.write(fileOut);
            fileOut.close();
        }
    }

    private void drawMseChart(NetworkData testData) {
        lineChart.getData().removeAll();
        List<DataEntry> dataEntries = testData.getDataEntries();
        final XYChart.Series<Integer, Double> series = new XYChart.Series<>();
        double[] predictedValues = network.getPredictedValues(testData, 1 / currentDataSet.getMaxOutput());
        for (int x = 0; x < dataEntries.size(); ++x) {
            series.getData().add(new XYChart.Data<>(x, dataEntries.get(x).getResult() - predictedValues[x]));
        }
        lineChart.getData().add(series);
    }

    private void drawComparisonChart(NetworkData testData) {
        lineChart2.getData().removeAll();
        List<DataEntry> dataEntries = testData.getDataEntries();
        final XYChart.Series<Integer, Double> realSeries = new XYChart.Series<>();
        final XYChart.Series<Integer, Double> predictedSeries = new XYChart.Series<>();
        double[] predictedValues = network.getPredictedValues(testData, 1 / currentDataSet.getMaxOutput());
        for (int x = 0; x < dataEntries.size(); ++x) {
            realSeries.getData().add(new XYChart.Data<>(x, dataEntries.get(x).getResult()));
            predictedSeries.getData().add(new XYChart.Data<>(x, predictedValues[x]));
        }
        realSeries.setName("real");
        predictedSeries.setName("predicted");
        lineChart2.getData().add(realSeries);
        lineChart2.getData().add(predictedSeries);
    }

    private void fillTable(NetworkData testData) {
        ObservableList<ErrorEntry> entries = FXCollections.observableArrayList();
        List<DataEntry> dataEntries = testData.getDataEntries();
        double[] predictedValues = network.getPredictedValues(testData, 1 / currentDataSet.getMaxOutput());
        double mapeDelta = calculateMape(testData, predictedValues);
        double rmseDelta = calculateRmse(testData, predictedValues);
        for (int x = 0; x < dataEntries.size(); ++x) {
            double realValue = dataEntries.get(x).getResult();
            double predictedValue = predictedValues[x];
            double delta = MathUtils.round(Math.abs(realValue - predictedValue),4);
            entries.add(new ErrorEntry(x, realValue, predictedValue, delta, MathUtils.round(Math.pow(delta, 2),4), mapeDelta, rmseDelta));
        }

        tableView.setItems(entries);

    }

    private double calculateMape(NetworkData testData, double[] predictedValues) {
        List<DataEntry> dataEntries = testData.getDataEntries();
        double deltaSum = 0.0;
        for (int i = 0; i < testData.getDataEntriesCount(); ++i) {
            deltaSum += Math.abs(dataEntries.get(i).getResult() - predictedValues[i]) / dataEntries.get(i).getResult();
        }
        return MathUtils.round(deltaSum/testData.getDataEntriesCount(), 4);
    }

    private double calculateRmse(NetworkData testData, double[] predictedValues) {
        List<DataEntry> dataEntries = testData.getDataEntries();
        double squaredDeltaSum = 0.0;
        for (int i = 0; i < testData.getDataEntriesCount(); ++i) {
            squaredDeltaSum += Math.pow(dataEntries.get(i).getResult() - predictedValues[i], 2);
        }
        return MathUtils.round(Math.sqrt(squaredDeltaSum/testData.getDataEntriesCount()), 4);
    }

    @Data
    public static class ErrorEntry {
        private final SimpleIntegerProperty point;
        private final SimpleDoubleProperty realValue;
        private final SimpleDoubleProperty predictedValue;
        private final SimpleDoubleProperty delta;
        private final SimpleDoubleProperty squaredDelta;
        private final SimpleDoubleProperty mapeDelta;
        private final SimpleDoubleProperty rmseDelta;

        public ErrorEntry(int point,
                          double realValue,
                          double predictedValue,
                          double delta,
                          double squaredDelta,
                          double mapeDelta,
                          double rmseDelta) {
            this.point = new SimpleIntegerProperty(point);
            this.realValue = new SimpleDoubleProperty(realValue);
            this.predictedValue = new SimpleDoubleProperty(predictedValue);
            this.delta = new SimpleDoubleProperty(delta);
            this.squaredDelta = new SimpleDoubleProperty(squaredDelta);
            this.mapeDelta = new SimpleDoubleProperty(mapeDelta);
            this.rmseDelta = new SimpleDoubleProperty(rmseDelta);
        }

        public int getPoint() {
            return point.get();
        }

        public void setPoint(int point) {
            this.point.set(point);
        }

        public double getRealValue() {
            return realValue.get();
        }

        public void setRealValue(double realValue) {
            this.realValue.set(realValue);
        }

        public double getPredictedValue() {
            return predictedValue.get();
        }

        public void setPredictedValue(double predictedValue) {
            this.predictedValue.set(predictedValue);
        }

        public double getDelta() {
            return delta.get();
        }

        public void setDelta(double delta) {
            this.delta.set(delta);
        }

        public double getSquaredDelta() {
            return squaredDelta.get();
        }

        public void setSquaredDelta(double squaredDelta) {
            this.squaredDelta.set(squaredDelta);
        }

        public double getMapeDelta() {
            return this.mapeDelta.get();
        }

        public void setMapeDelta(double mapeDelta) {
            this.mapeDelta.set(mapeDelta);
        }

        public double getRmseDelta() {
            return this.rmseDelta.get();
        }

        public void setRmseDelta(double rmseDelta) {
            this.rmseDelta.set(rmseDelta);
        }
    }
}
