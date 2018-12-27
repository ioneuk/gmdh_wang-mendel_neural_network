package org.gmdh.neuro.fuzzy.gmdh;

import lombok.Data;
import org.apache.commons.math3.util.Pair;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.config.NodeConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Data
public class GmdhLayer {

    private GmdhLayer prevLayer;
    private List<GmdhNode> nodes;

    private Map<GmdhNode, Pair<Integer, Integer>> inputValuesAssociation;

    public GmdhLayer(int inputNeuronsCount,GmdhLayer prevLayer, GmdhConfig gmdhConfig) {
        this.prevLayer = prevLayer;
        int nodeCount = calculateNodeCount(inputNeuronsCount);
        if(nodeCount > gmdhConfig.getMaxNeuronCountPerLayer()) {
            nodeCount = gmdhConfig.getMaxNeuronCountPerLayer();
        }
        nodes = new ArrayList<>(nodeCount);
        NodeConfig nodeConfig = createNodeConfigForLayer(gmdhConfig);
        for(int i = 0; i < nodeCount; ++i) {
            GmdhNode node = new GmdhNode(nodeConfig);
            nodes.add(node);
        }
    }

    public void train(NetworkData trainData) {
        nodes.forEach(node -> node.train(trainData));
    }

    public GmdhNode findNeuronWithLowestMse(NetworkData testData) {
        GmdhNode gmdhNodeWithLowestMse = nodes.get(0);
        double lowestMse = gmdhNodeWithLowestMse.calculateMse(testData);
        for (GmdhNode node : nodes) {
            double currentMse = node.calculateMse(testData);
            if(currentMse < lowestMse) {
                lowestMse = currentMse;
                gmdhNodeWithLowestMse = node;
            }
        }
        return gmdhNodeWithLowestMse;
    }

    public List<GmdhNode> findTopGmdhNodesWithLowestMse(int nodeCount) {
        return nodes.stream().sorted(Comparator.comparingDouble(GmdhNode::getLastMse)).limit(nodeCount).collect(Collectors.toList());
    }

    public int getNeuronCount() {
        return nodes.size();
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
