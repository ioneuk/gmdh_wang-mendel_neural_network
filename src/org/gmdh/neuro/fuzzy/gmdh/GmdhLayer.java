package org.gmdh.neuro.fuzzy.gmdh;

import lombok.Data;
import org.apache.commons.math3.util.Pair;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.config.NodeConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;

import java.util.*;
import java.util.stream.Collectors;

@Data
public class GmdhLayer {

    private String uid;

    private GmdhLayer prevLayer;
    private GmdhLayer nextLayer;
    private List<GmdhNode> nodes;

    private List<GmdhNode> workingNodes;

    private Map<GmdhNode, Pair<Integer, Integer>> inputValuesAssociation;

    public GmdhLayer(int inputNeuronsCount,GmdhLayer prevLayer, GmdhConfig gmdhConfig) {
        this.uid = UUID.randomUUID().toString();
        this.prevLayer = prevLayer;
        int nodeCount = calculateNodeCount(inputNeuronsCount);
        if(nodeCount > gmdhConfig.getMaxNeuronCountPerLayer()) {
            nodeCount = gmdhConfig.getMaxNeuronCountPerLayer();
        }
        nodes = new ArrayList<>(nodeCount);
        workingNodes = new ArrayList<>();

        NodeConfig nodeConfig = createNodeConfigForLayer(gmdhConfig);
        for(int i = 0; i < nodeCount; ++i) {
            GmdhNode node = new GmdhNode(nodeConfig);
            nodes.add(node);
        }
    }

    public void train(NetworkData trainData) {
        for (GmdhNode node : nodes) {
            Pair<Integer, Integer> inputs = inputValuesAssociation.get(node);
            node.train(trainData, inputs.getFirst(), inputs.getSecond());
        }
    }

    public double[] calculateOutput(DataEntry dataEntry) {
        double[] inputs = dataEntry.getRegressors();
        double[] result = new double[workingNodes.size()];
        int neuronsCount = inputs.length;

        int counter = 0;
        for(int i=0; i < neuronsCount - 1; ++i) {
            for (int j = i + 1; j < neuronsCount; ++j) {
                if(counter < workingNodes.size()) {
                    GmdhNode node = workingNodes.get(counter);
                    result[counter] = node.calculateOutput(inputs[i], inputs[j]);
                    counter++;
                }
            }
        }
        return result;
    }

    public GmdhNode findNeuronWithLowestMse(NetworkData testData) {
        GmdhNode gmdhNodeWithLowestMse = nodes.get(0);
        double lowestMse = Double.MAX_VALUE;
        for (GmdhNode node : nodes) {
            Pair<Integer, Integer> inputs = inputValuesAssociation.get(gmdhNodeWithLowestMse);
            double currentMse = node.calculateMse(testData, inputs.getFirst(), inputs.getSecond());
            if(currentMse < lowestMse) {
                lowestMse = currentMse;
                gmdhNodeWithLowestMse = node;
            }
        }

        nodes.sort(Comparator.comparingDouble(GmdhNode::getLastMse));
        return gmdhNodeWithLowestMse;
    }

    public List<GmdhNode> findTopGmdhNodesWithLowestMse(int nodeCount) {
        return nodes.stream().sorted(Comparator.comparingDouble(GmdhNode::getLastMse)).limit(nodeCount).collect(Collectors.toList());
    }

    public int getNeuronCount() {
        return nodes.size();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GmdhLayer gmdhLayer = (GmdhLayer) o;
        return Objects.equals(uid, gmdhLayer.uid);
    }

    @Override
    public int hashCode() {

        return Objects.hash(uid);
    }

    private int calculateNodeCount(int regressorsCount) {
        return regressorsCount * (regressorsCount - 1) / 2;
    }

    private NodeConfig createNodeConfigForLayer(GmdhConfig gmdhConfig) {
        NodeConfig nodeConfig = new NodeConfig();
        nodeConfig.setLearningRate(gmdhConfig.getLearningRate());
        nodeConfig.setMFunctionsPerInput(gmdhConfig.getMFunctionsPerInput());
        nodeConfig.setRegressorsCount(2);
        return nodeConfig;
    }
}
