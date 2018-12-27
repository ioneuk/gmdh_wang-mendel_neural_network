package org.gmdh.neuro.fuzzy.utils;

import java.util.Random;

public class GenerationUtils {

    public static void initWithConstant(double[][] matrix, double value) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = value;
            }
        }
    }

    public static void initWithConstant(double[] vector, double value) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = value;
        }
    }

    public static void initWithRandom(double[][] matrix) {
        Random random = new Random();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }
    }

    public static void initWithRandom(double[] vector) {
        Random random = new Random();
        for (int i = 0; i < vector.length; i++) {
            vector[i] = random.nextDouble();
        }
    }
}
