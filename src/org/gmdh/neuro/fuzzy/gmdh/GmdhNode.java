package org.gmdh.neuro.fuzzy.gmdh;

import lombok.Data;
import org.gmdh.neuro.fuzzy.gmdh.config.GmdhConfig;
import org.gmdh.neuro.fuzzy.gmdh.config.NodeConfig;
import org.gmdh.neuro.fuzzy.gmdh.data.DataEntry;
import org.gmdh.neuro.fuzzy.gmdh.data.NetworkData;
import org.gmdh.neuro.fuzzy.utils.ArrayUtils;
import org.gmdh.neuro.fuzzy.utils.GenerationUtils;
import org.gmdh.neuro.fuzzy.utils.MathUtils;

import java.util.List;
import java.util.Random;

@Data
public class GmdhNode {

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
        this.mFunctionsPerInput = config.getMFunctionsPerInput();
        this.learningRate = config.getLearningRate();
        this.regressorsCount = config.getRegressorsCount();
        sigma = new double[regressorsCount][config.getMFunctionsPerInput()];
        c = new double[regressorsCount][config.getMFunctionsPerInput()];
        w = new double[mFunctionsPerInput];

        Random random = new Random();
        GenerationUtils.initWithRandom(c);
        GenerationUtils.initWithConstant(sigma, 0.5);
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

    public double calculateMse() {
        return 0;
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
        for (int i = 0; i < K; ++i) {
            D[i] = trainData.getDataEntryWithNumber(i).getResult();
        }

        double[][] pseudoPV = MathUtils.pseudoInverse(PV);
        w = MathUtils.multiply(pseudoPV, D);




        for (DataEntry data : trainData.getDataEntries()) {
            double[][] nextC = reuseC;
            double[][] nextSigma = reuseSigma;
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
                    nextC[i][j] = c[i][j] - learningRate * dE_dc(i, j, output, expected, data.getRegressors());
                    nextSigma[i][j] = sigma[i][j] - learningRate * dE_dSigma(i, j, output, expected, data.getRegressors());
                }
            }
            ArrayUtils.copy(reuseC, c);
            ArrayUtils.copy(reuseSigma, sigma);
        }
    }

    private double l(int row, double x1, double x2) {
        double l = 1;
        for (int i = 0; i < 2; i++) {
            l = MathUtils.gauss(x1, c[row][i], sigma[row][i]) * MathUtils.gauss(x2, c[row][i], sigma[row][i]);
        }
        return l;
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
        return dE_dZ0(output, expected) * dZ0_dZk() * dZk_dC(k, j, x);
    }

    private double dE_dSigma(int k, int j, double output, double expected, double[] x) {
        return dE_dZ0(output, expected) * dZ0_dZk() * dZk_dSigma(k, j ,x);
    }

    private double dE_dZ0(double output, double expected) {
        return output - expected;
    }

    private double dZ0_dZk() {
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
}
