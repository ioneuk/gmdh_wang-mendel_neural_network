package org.gmdh.neuro.fuzzy.gmdh;

import lombok.Data;
import org.gmdh.neuro.fuzzy.gmdh.config.NodeConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.utils.GenerationUtils;
import org.gmdh.neuro.fuzzy.utils.MathUtils;
import org.gmdh.neuro.fuzzy.utils.NormalizationUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

@Data
public class GmdhNode {
    private static final int iterationCount = 50;

    private String uid;
    private double slope;
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
        this.slope = config.getSlope();
        this.mFunctionsPerInput = config.getMFunctionsPerInput();
        this.learningRate = config.getLearningRate();
        this.regressorsCount = config.getRegressorsCount();
        sigma = new double[mFunctionsPerInput][regressorsCount];
        c = new double[mFunctionsPerInput][regressorsCount];
        w = new double[mFunctionsPerInput];

        reuseC = new double[mFunctionsPerInput][regressorsCount];
        reuseSigma = new double[mFunctionsPerInput][regressorsCount];
        lCache = new double[mFunctionsPerInput];

        inputNodes = new ArrayList<>();
        outputNodes = new ArrayList<>();

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

    public double calculateMse(NetworkData testData, int firstInput, int secondInput) {
        double squaredError = 0;

        for (DataEntry dataEntry : testData.getDataEntries()) {
            double x1 = dataEntry.getInputByColumnNumber(firstInput);
            double x2 = dataEntry.getInputByColumnNumber(secondInput);
            squaredError += Math.pow(calculateOutput(x1, x2) - dataEntry.getResult(), 2);
        }

        lastMse = squaredError / 2;
        return lastMse;
    }

    public void train(NetworkData trainData, int firstInput, int secondInput) {
        int K = trainData.getDataEntriesCount();
        double[][] PV = new double[K][mFunctionsPerInput];
        /*for (int i = 0; i < K; i++) {
            DataEntry dataEntry = trainData.getDataEntryWithNumber(i);
            double x1 = dataEntry.getInputByColumnNumber(firstInput);
            double x2 = dataEntry.getInputByColumnNumber(secondInput);

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
        w = MathUtils.multiply(pseudoPV, D);*/

        for(int b = 0; b < iterationCount; ++b) {


            for (int i = 0; i < K; i++) {
                DataEntry dataEntry = trainData.getDataEntryWithNumber(i);
                double x1 = dataEntry.getInputByColumnNumber(firstInput);
                double x2 = dataEntry.getInputByColumnNumber(secondInput);

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


            double[][] accC = new double[mFunctionsPerInput][regressorsCount];
            double[][] accSigma = new double[mFunctionsPerInput][regressorsCount];
            for (DataEntry data : trainData.getDataEntries()) {
                double x1 = data.getInputByColumnNumber(firstInput);
                double x2 = data.getInputByColumnNumber(secondInput);
                mCache = m(x1, x2);
                for (int i = 0; i < mFunctionsPerInput; i++) {
                    lCache[i] = l(i, x1, x2);
                }
                double output = calculateOutput(x1, x2);
                double expected = data.getResult();
                for (int i = 0; i < mFunctionsPerInput; i++) {
                    for (int j = 0; j < regressorsCount; j++) {
                        accC[i][j] += dE_dc(i, j, output, expected, new double[]{x1, x2});
                        accSigma[i][j] += dE_dSigma(i, j, output, expected, new double[]{x1, x2});
                    }
                }
            }

            for(int i = 0; i < mFunctionsPerInput; ++i) {
                for(int j = 0; j < regressorsCount; ++j) {
                    c[i][j] -= learningRate * accC[i][j] / K;
                    sigma[i][j] -= learningRate * accSigma[i][j] / K;
                }
            }
        }
    }

    private double l(int row, double x1, double x2) {
        double result = MathUtils.gauss(x1, c[row][0], sigma[row][0]) * MathUtils.gauss(x2, c[row][1], sigma[row][1]);
        return result;
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
        return dE_dZ0(output, expected) * dZ0_dZk(k, x) * dZk_dC(k, j, x) * d_Sigmoid(output);
    }

    private double dE_dSigma(int k, int j, double output, double expected, double[] x) {
        return dE_dZ0(output, expected) * dZ0_dZk(k, x) * dZk_dSigma(k, j ,x) * d_Sigmoid(output);
    }

    private double dE_dW(double output, double expected) {
        return dE_dZ0(output, expected);
    }

    private double dE_dZ0(double output, double expected) {
        return output - expected;
    }

    private double d_Sigmoid(double x) {
        return NormalizationUtils.sigmoid(x, slope) * (1 - NormalizationUtils.sigmoid(x, slope)) * slope;
    }

    private double dZ0_dZk(int k, double[] x) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 0; i < mFunctionsPerInput; i++) {
            double res = l(i, x[0], x[1]);
            numerator += res * w[i];
            denominator += res;
        }
        return l(k, x[0], x[1]) * (w[k] * denominator - numerator) / Math.pow(denominator, 2);
    }

    private double dZ0_dWk() {
        return 1;
    }

    private double dZk_dC(int k, int j, double[] x) {
        return (x[j] - c[k][j]) / (Math.pow(sigma[k][j], 2));
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
