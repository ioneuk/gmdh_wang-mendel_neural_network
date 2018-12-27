package org.gmdh.neuro.fuzzy.gmdh;

import lombok.Data;
import org.gmdh.neuro.fuzzy.gmdh.config.NodeConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.utils.GenerationUtils;
import org.gmdh.neuro.fuzzy.utils.MathUtils;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.UUID;

@Data
public class GmdhNode {

    private String uid;
    /**
     * non-linear weghts
     */
    private double[][] c;
    private double[][] sigma;

    /**
     * linear weight
     */
    private double[] w;

    /**
     * The layer which this node belongs to
     */
    private GmdhLayer gmdhLayer;

    private int regressorsCount;
    private int mFunctionsPerInput;
    private double learningRate;

    private List<GmdhNode> inputNodes;
    private List<GmdhNode> outputNodes;

    private double lastMse;
    private double lastOutput;

    private double[][] reuseC;
    private double[][] reuseSigma;
    private double[] lCache;
    private double mCache;

    public GmdhNode(NodeConfig config) {
        this.uid = UUID.randomUUID().toString();
        this.mFunctionsPerInput = config.getMFunctionsPerInput();
        this.learningRate = config.getLearningRate();
        this.regressorsCount = config.getRegressorsCount();
        sigma = new double[mFunctionsPerInput][regressorsCount];
        c = new double[mFunctionsPerInput][regressorsCount];
        w = new double[mFunctionsPerInput];

        reuseC = new double[mFunctionsPerInput][regressorsCount];
        reuseSigma = new double[mFunctionsPerInput][regressorsCount];
        lCache = new double[mFunctionsPerInput];

        GenerationUtils.initWithRandom(c);
        GenerationUtils.initWithConstant(sigma, 1);
        GenerationUtils.initWithRandom(w);
    }

    public double calculateOutput(double x1, double x2) {
        double numerator = 0;
        double denominator = 0;
        for (int i = 0; i < mFunctionsPerInput; i++) {
            double res = l(i, x1, x2);
            numerator += res * w[i];
            denominator += res;
        }
        return numerator / denominator;
    }

    public double calculateMse(NetworkData testData) {
        double squaredError = 0;

        for (DataEntry dataEntry : testData.getDataEntries()) {
            double x1 = dataEntry.getInputByColumnNumber(0);
            double x2 = dataEntry.getInputByColumnNumber(1);
            squaredError += Math.pow(calculateOutput(x1, x2) - dataEntry.getResult(), 2);
        }

        lastMse = squaredError / testData.getDataEntries().size();
        return lastMse;
    }

    public void train(NetworkData trainData) {
        int K = trainData.getDataEntries().size();
        double[][] PV = new double[K][mFunctionsPerInput];
        for (int i = 0; i < K; i++) {
            DataEntry dataEntry = trainData.getDataEntryWithNumber(i);
            double x1 = dataEntry.getInputByColumnNumber(0);
            double x2 = dataEntry.getInputByColumnNumber(1);

            double denominator = 0;
            for (int j = 0; j < mFunctionsPerInput; j++) {
                //denominator += l(j, x);
                denominator += l(j, x1, x2);
            }
            for (int j = 0; j < mFunctionsPerInput; j++) {
                PV[i][j] = l(j, x1, x2) / denominator;
            }
        }

        double[] D = new double[K];
        for (int i = 0; i < K; i++) {
            D[i] = trainData.getDataEntries().get(i).getResult();
        }

        double[][] pseudoPV = MathUtils.pseudoInverse(PV);
        w = MathUtils.multiply(pseudoPV, D);

        for (DataEntry data : trainData.getDataEntries()) {
            double x1 = data.getInputByColumnNumber(0);
            double x2 = data.getInputByColumnNumber(1);
            mCache = m(x1, x2);
            for (int i = 0; i < mFunctionsPerInput; i++) {
                lCache[i] = l(i, x1, x2);
            }
            double output = calculateOutput(x1, x2);
            double expected = data.getResult();
            for (int i = 0; i < mFunctionsPerInput; i++) {
                for (int j = 0; j < regressorsCount; j++) {
                    c[i][j] -= learningRate * dE_dc(i, j, output, expected, data.getRegressors());
                    sigma[i][j] -= learningRate * dE_dSigma(i, j, output, expected, data.getRegressors());
                }
            }
        }
    }

    private double l(int row, double x1, double x2) {
        return MathUtils.gauss(x1, c[row][0], sigma[row][0]) * MathUtils.gauss(x2, c[row][1], sigma[row][1]);
    }

    private double m(double x1, double x2) {
        double m = 0;
        for (int row = 0; row < mFunctionsPerInput; row++) {
            m += l(row, x1, x2);
        }
        return m;
    }

    private double m(int i, int j, double x) {
        return MathUtils.gauss(x, c[i][j], sigma[i][j]);
    }


    private double dE_dc(int k, int j, double output, double expected, double[] x) {
        return dE_dZ0(output, expected) * dZ0_dZk(k, j, x) * dZk_dC(k, j, x);
    }

    private double dE_dSigma(int k, int j, double output, double expected, double[] x) {
        return dE_dZ0(output, expected) * dZ0_dZk(k, j, x) * dZk_dSigma(k, j ,x);
    }

    private double dE_dW(double output, double expected) {
        return dE_dZ0(output, expected);
    }

    private double dE_dZ0(double output, double expected) {
        return output - expected;
    }

    private double dZ0_dZk(int k, int j, double[] x) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 0; i < mFunctionsPerInput; i++) {
            double res = l(i, x[0], x[1]);
            numerator += res * w[i];
            denominator += res;
        }
        return (w[k] * denominator - numerator) / Math.pow(denominator, 2);
    }

    private double dZ0_dWk() {
        return 1;
    }

    private double dZk_dC(int k, int j, double[] x) {
        return (x[j] - c[k][j]) / (2 * Math.pow(sigma[k][j], 2));
    }

    private double dZk_dSigma(int k, int j, double[] x) {
        return Math.pow(x[j] - c[k][j], 2) / Math.pow(sigma[k][j], 3);
    }

    private double kronekerDelta(int i, int j) {
        return i == j ? 1 : 0;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GmdhNode gmdhNode = (GmdhNode) o;
        return Objects.equals(uid, gmdhNode.uid);
    }

    @Override
    public int hashCode() {
        return Objects.hash(uid);
    }
}
