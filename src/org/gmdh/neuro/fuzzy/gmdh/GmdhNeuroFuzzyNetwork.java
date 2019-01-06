package org.gmdh.neuro.fuzzy.gmdh;

import org.apache.commons.math3.util.Pair;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.utils.NormalizationUtils;

import java.util.*;
import java.util.stream.Collectors;

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
        //double currentMse = currentNodeWithLowestMse.calculateMse(testData);
        double currentMse = currentNodeWithLowestMse.getLastMse();

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
            layers.add(currentLayer);

            currentLayer.train(currentTrainData);
            currentNodeWithLowestMse = currentLayer.findNeuronWithLowestMse(currentTestData);
            trainOutputDataStorage.put(currentLayer, composeOutputData(currentLayer, currentTrainData));
            testOutputDataStorage.put(currentLayer, composeOutputData(currentLayer, currentTestData));

            currentMse = currentNodeWithLowestMse.getLastMse();
            prevLayer.setNextLayer(currentLayer);
            prevLayer = currentLayer;
        } while (currentMse < prevMse);
        layers.remove(layers.size()-1);
        setupOptimalNetworkStructure(bestNode);
        System.out.println("HO HO HO!!!");
    }

    public double[] getPredictedValues(NetworkData testData) {
        List<DataEntry> dataEntries = testData.getDataEntries();
        double[] result = new double[dataEntries.size()];

        for (int i = 0; i < dataEntries.size(); i++) {
            DataEntry dataEntry = dataEntries.get(i);
            DataEntry modifiedDataEntry = dataEntry.clone();
            GmdhLayer currentLayer = layers.get(0);
            double[] currentOutput;
            do {
                currentOutput = currentLayer.calculateOutput(modifiedDataEntry);
                modifiedDataEntry.setRegressors(currentOutput);
                currentLayer = currentLayer.getNextLayer();
            } while (currentLayer.getNextLayer() != null);
            result[i] = currentOutput[0];
        }
        return result;
    }

    private void setupOptimalNetworkStructure(GmdhNode targetNode) {
        GmdhLayer lastLayer = layers.get(layers.size() - 1);
        lastLayer.setWorkingNodes(Arrays.asList(targetNode));
        List<GmdhNode> currentNodes = Arrays.asList(targetNode);
        for(int i = layers.size()-2 ; i >= 0; --i) {
            GmdhLayer currentLayer = layers.get(i);
            currentNodes = currentNodes.stream().flatMap(node -> node.getInputNodes().stream()).distinct().collect(Collectors.toList());
            currentLayer.setWorkingNodes(currentNodes);
        }
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

        for (GmdhNode node : gmdhNodePairMap.keySet()) {
            Pair<Integer,Integer> inputNodes = gmdhNodePairMap.get(node);
            GmdhNode firstNode = baseLayer.getNodes().get(inputNodes.getFirst());
            GmdhNode secondNode = baseLayer.getNodes().get(inputNodes.getSecond());
            node.setInputNodes(Arrays.asList(firstNode, secondNode));
            firstNode.getOutputNodes().add(node);
        }

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
                    GmdhNode node = nodes.get(counter);
                    inputValueAssociation.put(node, new Pair<>(i, j));
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
}
