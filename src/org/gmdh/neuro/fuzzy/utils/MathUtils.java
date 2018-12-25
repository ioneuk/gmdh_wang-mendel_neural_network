package org.gmdh.neuro.fuzzy.utils;

import org.ejml.simple.SimpleMatrix;

public class MathUtils {

    public static double gauss(double x, double c, double sigma) {
        //return 1 / (1 + Math.pow((x - c) / 2 * sigma * sigma, 2 * b));
        return Math.exp(-1 * Math.pow(x-c, 2) / (2 * Math.pow(sigma, 2)));
    }

    public static double[][] pseudoInverse(double[][] m) {
        SimpleMatrix simpleMatrix = new SimpleMatrix(m);
        SimpleMatrix inverse = simpleMatrix.pseudoInverse();
        return convert(inverse);
    }

    private static double[][] convert(SimpleMatrix matrix) {
        double[][] result = new double[matrix.numRows()][matrix.numCols()];
        for (int i = 0; i < matrix.numRows(); i++) {
            for (int j = 0; j < matrix.numCols(); j++) {
                result[i][j] = matrix.get(i, j);
            }
        }
        return result;
    }

    public static double[] multiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        if (cols != vector.length) throw new IllegalArgumentException();

        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
}
