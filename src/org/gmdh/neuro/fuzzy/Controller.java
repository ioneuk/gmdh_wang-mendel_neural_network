package org.gmdh.neuro.fuzzy;

import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import org.gmdh.neuro.fuzzy.gmdh.GmdhNeuroFuzzyNetwork;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.DataSet;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.reader.DataReader;

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

    private DataReader dataReader = new DataReader();
    private GmdhNeuroFuzzyNetwork network;

    @FXML
    public void initialize() {
        SpinnerValueFactory.IntegerSpinnerValueFactory mFunctionsValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(0, 10);
        mFunctionsPerInput.setValueFactory(mFunctionsValueFactory);
        SpinnerValueFactory.DoubleSpinnerValueFactory learningRateValueFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 3.0, 0.5, 0.25);
        learningRate.setValueFactory(learningRateValueFactory);
        SpinnerValueFactory.IntegerSpinnerValueFactory bestNeuronCountValueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 1000);
        bestNeuronCount.setValueFactory(bestNeuronCountValueFactory);
    }

    @FXML
    public void onTrainAction() {
        GmdhConfig gmdhConfig = new GmdhConfig();
        gmdhConfig.setMaxNeuronCountPerLayer(bestNeuronCount.getValue());
        gmdhConfig.setLearningRate(learningRate.getValue());
        gmdhConfig.setLearnTestRatio(learnTestRatio.getValue());
        gmdhConfig.setRegressorsCount(2);
        gmdhConfig.setMFunctionsPerInput(mFunctionsPerInput.getValue());
        DataSet dataSet = dataReader.loadDataSet((int) gmdhConfig.getLearnTestRatio());
        network = new GmdhNeuroFuzzyNetwork(gmdhConfig);
        network.buildGmdhStructure(dataSet.getTrainData(), dataSet.getTestData());
        drawMseChart(dataSet.getTestData());
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
}
