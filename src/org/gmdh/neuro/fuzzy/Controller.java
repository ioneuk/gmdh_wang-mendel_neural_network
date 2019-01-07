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
import org.gmdh.neuro.fuzzy.gmdh.GmdhNeuroFuzzyNetwork;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.DataSet;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.reader.DataReader;

import java.io.File;
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

    private FileChooser fileChooser;
    private GmdhConfig gmdhConfig;
    private DataReader dataReader;
    private DataSet currentDataSet;
    private GmdhNeuroFuzzyNetwork network;

    @FXML
    public void initialize() {
        SpinnerValueFactory.IntegerSpinnerValueFactory mFunctionsValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(0, 10, 3);
        mFunctionsPerInput.setValueFactory(mFunctionsValueFactory);
        SpinnerValueFactory.DoubleSpinnerValueFactory learningRateValueFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 3.0, 0.5, 0.25);
        learningRate.setValueFactory(learningRateValueFactory);
        SpinnerValueFactory.IntegerSpinnerValueFactory bestNeuronCountValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 1000, 200);
        bestNeuronCount.setValueFactory(bestNeuronCountValueFactory);
        SpinnerValueFactory.IntegerSpinnerValueFactory inputNumberValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 20, 4);
        inputNumber.setValueFactory(inputNumberValueFactory);

        pointTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Integer>("point"));
        realValueTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("realValue"));
        predictedValueTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("predictedValue"));
        deltaTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("delta"));
        squaredDeltaTab.setCellValueFactory(new PropertyValueFactory<ErrorEntry, Double>("squaredDelta"));

        dataReader = new DataReader();
        fileChooser = new FileChooser();
        tableView.setEditable(true);
    }

    @FXML
    public void onTrainAction() {
        GmdhConfig gmdhConfig = new GmdhConfig();
        gmdhConfig.setMaxNeuronCountPerLayer(bestNeuronCount.getValue());
        gmdhConfig.setLearningRate(learningRate.getValue());
        gmdhConfig.setLearnTestRatio(learnTestRatio.getValue());
        gmdhConfig.setRegressorsCount(2);
        gmdhConfig.setMFunctionsPerInput(mFunctionsPerInput.getValue());
        network = new GmdhNeuroFuzzyNetwork(gmdhConfig);
        NetworkData testData = currentDataSet.getTestData();
        network.buildGmdhStructure(currentDataSet.getTrainData(), testData);

        drawMseChart(testData);
        drawComparisonChart(testData);
        fillTable(testData);
    }

    @FXML
    public void onLoadDataAction() {
        File file = fileChooser.showOpenDialog(new Stage());
        if (file != null) {
            currentDataSet = dataReader.loadDataSet(file, (int) learnTestRatio.getValue(), inputNumber.getValue());
        }
    }

    private void drawMseChart(NetworkData testData) {
        List<DataEntry> dataEntries = testData.getDataEntries();
        final XYChart.Series<Integer, Double> series = new XYChart.Series<>();
        double[] predictedValues = network.getPredictedValues(testData);
        for (int x = 0; x < dataEntries.size(); ++x) {
            series.getData().add(new XYChart.Data<>(x, dataEntries.get(x).getResult() - predictedValues[x]));
        }
        lineChart.getData().add(series);
    }

    private void drawComparisonChart(NetworkData testData) {
        List<DataEntry> dataEntries = testData.getDataEntries();
        final XYChart.Series<Integer, Double> realSeries = new XYChart.Series<>();
        final XYChart.Series<Integer, Double> predictedSeries = new XYChart.Series<>();
        double[] predictedValues = network.getPredictedValues(testData);
        for (int x = 0; x < dataEntries.size(); ++x) {
            realSeries.getData().add(new XYChart.Data<>(x, dataEntries.get(x).getResult()));
            predictedSeries.getData().add(new XYChart.Data<>(x, predictedValues[x]));
        }
        lineChart2.getData().add(realSeries);
        lineChart2.getData().add(predictedSeries);
    }

    private void fillTable(NetworkData testData) {
        ObservableList<ErrorEntry> entries = FXCollections.observableArrayList();
        List<DataEntry> dataEntries = testData.getDataEntries();
        double[] predictedValues = network.getPredictedValues(testData);
        for (int x = 0; x < dataEntries.size(); ++x) {
            double realValue = dataEntries.get(x).getResult();
            double predictedValue = predictedValues[x];
            double delta = Math.abs(realValue - predictedValue);
            entries.add(new ErrorEntry(x, realValue, predictedValue, delta, Math.pow(delta, 2)));
        }
        tableView.setItems(entries);
    }

    @Data
    public static class ErrorEntry {
        private final SimpleIntegerProperty point;
        private final SimpleDoubleProperty realValue;
        private final SimpleDoubleProperty predictedValue;
        private final SimpleDoubleProperty delta;
        private final SimpleDoubleProperty squaredDelta;

        public ErrorEntry(int point, double realValue, double predictedValue, double delta, double squaredDelta) {
            this.point = new SimpleIntegerProperty(point);
            this.realValue = new SimpleDoubleProperty(realValue);
            this.predictedValue = new SimpleDoubleProperty(predictedValue);
            this.delta = new SimpleDoubleProperty(delta);
            this.squaredDelta = new SimpleDoubleProperty(squaredDelta);
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
    }
}
