package org.gmdh.neuro.fuzzy.gmdh;

import org.apache.commons.math3.util.Pair;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GmdhNeuroFuzzyNetwork {

    private static final Double EPS = 0.001;

    private GmdhConfig gmdhConfig;
    private List<Double> mseForLayers;
    private List<GmdhLayer> layers;
    private List<GmdhLayer> optimalLayers;

    Map<GmdhLayer, NetworkData> trainOutputDataStorage;
    Map<GmdhLayer, NetworkData> testOutputDataStorage;

    public GmdhNeuroFuzzyNetwork(GmdhConfig config) {
        this.gmdhConfig = config;
        mseForLayers = new ArrayList<>();
        layers = new ArrayList<>();
    }

    public void buildGmdhStructure(NetworkData trainData, NetworkData testData) {
        layers.clear();
        GmdhLayer firstLayer = createFirstNeuronLayer(trainData);
        layers.add(firstLayer);
        firstLayer.train(trainData);
        trainOutputDataStorage.put(firstLayer, composeOutputData(firstLayer, trainData));
        testOutputDataStorage.put(firstLayer, composeOutputData(firstLayer, testData));

        GmdhLayer prevLayer = firstLayer;
        double prevMse;
        double currentMse = firstLayer.findNeuronWithLowestMse(testData).calculateMse();
        NetworkData currentTrainData;
        NetworkData currentTestData;
        GmdhLayer currentLayer;
        do {
            prevMse = currentMse;
            currentLayer = createSequentialNeuronLayer(prevLayer);
            currentTrainData = trainOutputDataStorage.get(prevLayer);
            currentTestData = testOutputDataStorage.get(prevLayer);

            currentLayer.train(currentTrainData);
            GmdhNode bestNeuron = currentLayer.findNeuronWithLowestMse(currentTestData);

            currentMse = bestNeuron.getLastMse();
            prevLayer = currentLayer;
        } while (currentMse < prevMse || currentMse > EPS);
    }

    private GmdhLayer createFirstNeuronLayer(NetworkData trainData) {
        GmdhLayer gmdhLayer = new GmdhLayer(trainData.getDataEntries().get(0).getRegressors().length, gmdhConfig);
        Map<GmdhNode, Pair<Integer, Integer>> gmdhNodePairMap = composeAssociationBetweenInputsAndNodes(gmdhLayer);

        gmdhLayer.setInputValuesAssociation(gmdhNodePairMap);
        return gmdhLayer;
    }

    private GmdhLayer createSequentialNeuronLayer(GmdhLayer baseLayer) {
        GmdhLayer gmdhLayer = new GmdhLayer(baseLayer.getNeuronCount(), gmdhConfig);
        Map<GmdhNode, Pair<Integer, Integer>> gmdhNodePairMap = composeAssociationBetweenInputsAndNodes(gmdhLayer);

        gmdhLayer.setInputValuesAssociation(gmdhNodePairMap);
        return gmdhLayer;
    }

    private Map<GmdhNode, Pair<Integer, Integer>> composeAssociationBetweenInputsAndNodes(GmdhLayer gmdhLayer) {
        Map<GmdhNode, Pair<Integer, Integer>> inputValueAssociation = new HashMap<>();
        List<GmdhNode> nodes = gmdhLayer.getNodes();

        for(int i=0; i < nodes.size() - 1; ++i) {
            for (int j = i + 1; j < nodes.size(); ++i) {
                inputValueAssociation.put(nodes.get(i), new Pair<>(i, j));
            }
        }

        return inputValueAssociation;
    }

    private NetworkData composeOutputData(GmdhLayer baseLayer, NetworkData data) {
        NetworkData networkData = new NetworkData();
        List<GmdhNode> nodes = baseLayer.getNodes();
        for (DataEntry dataEntry : data.getDataEntries()) {
            DataEntry newDataEntry = new DataEntry();
            double[] inputs = new double[nodes.size()];
            for (int i = 0; i < nodes.size(); ++i) {
                Pair<Integer, Integer> regressorIds = baseLayer.getInputValuesAssociation().get(nodes.get(i));
                double firstInput = dataEntry.getInputByColumnNumber(regressorIds.getFirst());
                double secondInput = dataEntry.getInputByColumnNumber(regressorIds.getSecond());
                double output = nodes.get(i).calculateOutput(firstInput, secondInput);
                inputs[i] = output;
            }
            newDataEntry.setRegressors(inputs);
            newDataEntry.setResult(dataEntry.getResult());
            networkData.getDataEntries().add(newDataEntry);
        }
        return networkData;
    }

    /*private NetworkData composeOutputTestData(GmdhLayer baseLayer, NetworkData testData) {
        NetworkData networkData = new NetworkData();
        List<GmdhNode> nodes = baseLayer.getNodes();
        for (DataEntry dataEntry : testData.getDataEntries()) {
            DataEntry newDataEntry = new DataEntry();
            double[] inputs = new double[baseLayer.getNodes().size()];
            for (int i = 0; i < nodes.size(); ++i) {
                Pair<Integer, Integer> regressorIds = baseLayer.getInputValuesAssociation().get(nodes.get(i));
                double firstInput = dataEntry.getInputByColumnNumber(regressorIds.getFirst());
                double secondInput = dataEntry.getInputByColumnNumber(regressorIds.getSecond());
                double output = nodes.get(i).calculateOutput(firstInput, secondInput);
                inputs[1] = output;
            }
            newDataEntry.setRegressors(inputs);
            newDataEntry.setResult(dataEntry.getResult());
            networkData.getDataEntries().add(newDataEntry);
        }
        return networkData;
    }*/
}
