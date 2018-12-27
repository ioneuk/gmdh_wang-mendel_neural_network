package org.gmdh.neuro.fuzzy.gmdh;

import org.apache.commons.math3.util.Pair;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.utils.NormalizationUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GmdhNeuroFuzzyNetwork {

    private GmdhConfig gmdhConfig;
    private List<GmdhLayer> layers;

    Map<GmdhLayer, NetworkData> trainOutputDataStorage = new HashMap<>();
    Map<GmdhLayer, NetworkData> testOutputDataStorage = new HashMap<>();

    public GmdhNeuroFuzzyNetwork(GmdhConfig config) {
        this.gmdhConfig = config;
        layers = new ArrayList<>();
    }

    public void buildGmdhStructure(NetworkData trainData, NetworkData testData) {
        layers.clear();
        GmdhLayer firstLayer = createFirstNeuronLayer(trainData);
        layers.add(firstLayer);
        firstLayer.train(trainData);
        GmdhNode currentNodeWithLowestMse = firstLayer.findNeuronWithLowestMse(testData);
        double currentMse = currentNodeWithLowestMse.calculateMse(testData);

        trainOutputDataStorage.put(firstLayer, composeOutputData(firstLayer, trainData));
        testOutputDataStorage.put(firstLayer, composeOutputData(firstLayer, testData));

        GmdhLayer prevLayer = firstLayer;
        GmdhNode bestNode;
        double prevMse;
        NetworkData currentTrainData;
        NetworkData currentTestData;
        GmdhLayer currentLayer;
        do {
            prevMse = currentMse;
            bestNode = currentNodeWithLowestMse;
            currentTrainData = trainOutputDataStorage.get(prevLayer);
            currentTestData = testOutputDataStorage.get(prevLayer);
            currentLayer = createSequentialNeuronLayer(prevLayer, currentTrainData);

            currentLayer.train(currentTrainData);
            currentNodeWithLowestMse = currentLayer.findNeuronWithLowestMse(currentTestData);
            trainOutputDataStorage.put(currentLayer, composeOutputData(currentLayer, currentTrainData));
            testOutputDataStorage.put(currentLayer, composeOutputData(currentLayer, currentTestData));

            currentMse = currentNodeWithLowestMse.calculateMse(currentTestData);
            prevLayer = currentLayer;
        } while (currentMse < prevMse);
        System.out.println("WOW WOW WOW");
    }

    private GmdhLayer createFirstNeuronLayer(NetworkData trainData) {
        GmdhLayer gmdhLayer = new GmdhLayer(trainData.getDataEntries().get(0).getRegressors().length, null, gmdhConfig);
        Map<GmdhNode, Pair<Integer, Integer>> gmdhNodePairMap = composeAssociationBetweenInputsAndNodes(gmdhLayer, trainData.getDataEntries().get(0).getRegressors().length);

        gmdhLayer.setInputValuesAssociation(gmdhNodePairMap);
        return gmdhLayer;
    }

    private GmdhLayer createSequentialNeuronLayer(GmdhLayer baseLayer, NetworkData trainData) {
        GmdhLayer gmdhLayer = new GmdhLayer(baseLayer.getNeuronCount(), baseLayer, gmdhConfig);
        Map<GmdhNode, Pair<Integer, Integer>> gmdhNodePairMap = composeAssociationBetweenInputsAndNodes(gmdhLayer, trainData.getDataEntries().get(0).getRegressors().length);

        gmdhLayer.setInputValuesAssociation(gmdhNodePairMap);
        return gmdhLayer;
    }

    private Map<GmdhNode, Pair<Integer, Integer>> composeAssociationBetweenInputsAndNodes(GmdhLayer gmdhLayer, int regressorsCount) {
        Map<GmdhNode, Pair<Integer, Integer>> inputValueAssociation = new HashMap<>();
        List<GmdhNode> nodes = gmdhLayer.getNodes();

        int counter = 0;
        for(int i=0; i < regressorsCount - 1; ++i) {
            for (int j = i + 1; j < regressorsCount; ++j) {
                if(counter < nodes.size()) {
                    inputValueAssociation.put(nodes.get(counter), new Pair<>(i, j));
                    counter++;
                }
            }
        }

        return inputValueAssociation;
    }

    private NetworkData composeOutputData(GmdhLayer baseLayer, NetworkData data) {
        NetworkData networkData = new NetworkData();
        networkData.setDataEntries(new ArrayList<>());
        int totalOutputsCount = calculateTotalOutputsCount(baseLayer.getNodes().size());

        List<GmdhNode> nodes = baseLayer.findTopGmdhNodesWithLowestMse(totalOutputsCount);

        for (DataEntry dataEntry : data.getDataEntries()) {
            DataEntry newDataEntry = new DataEntry();
            double[] inputs = new double[totalOutputsCount];
            for (int i = 0; i < totalOutputsCount; ++i) {
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
        NormalizationUtils.normalizeDataSet(networkData);
        return networkData;
    }

    private int calculateTotalOutputsCount(int nodesCount) {
        return nodesCount < gmdhConfig.getMaxNeuronCountPerLayer() ? nodesCount : gmdhConfig.getMaxNeuronCountPerLayer();
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
