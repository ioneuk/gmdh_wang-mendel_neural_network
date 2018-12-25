package org.gmdh.neuro.fuzzy.gmdh;

import lombok.Data;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.apache.commons.math3.util.Pair;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.config.NodeConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Data
public class GmdhLayer {

    private GmdhLayer prevLayer;
    private List<GmdhNode> nodes;

    private Map<GmdhNode, Pair<Integer, Integer>> inputValuesAssociation;

    public GmdhLayer(int inputNeuronsCount, GmdhConfig gmdhConfig) {
        int nodeCount = calculateNodeCount(inputNeuronsCount);
        nodes = new ArrayList<>(nodeCount);

    }

    public void train(NetworkData trainData) {
        nodes.forEach(node -> node.train(trainData));
    }

    public GmdhNode findNeuronWithLowestMse(NetworkData testData) {
        nodes.stream();
        return null;
    }

    public int getNeuronCount() {
        return nodes.size();
    }

    private int calculateNodeCount(int regressorsCount) {
        return (int)CombinatoricsUtils.factorial(regressorsCount)/(int)(2 * CombinatoricsUtils.factorial(regressorsCount - 2));
    }

    private NodeConfig createNodeConfigForLayer(GmdhConfig gmdhConfig) {
        NodeConfig nodeConfig = new NodeConfig();
        nodeConfig.setLearningRate(gmdhConfig.getLearningRate());
        nodeConfig.setMFunctionsPerInput(gmdhConfig.getMFunctionsPerInput());
        nodeConfig.setRegressorsCount(2);
        return nodeConfig;
    }
}
