package org.gmdh.neuro.fuzzy.utils;

public class ArrayUtils {

    public static void copy(double[][] src, double[][] dst) {
        if (src.length != dst.length) throw new IllegalArgumentException();
        for (int i = 0; i < src.length; i++) {
            if (src[0].length != dst[0].length) throw new IllegalArgumentException();
            for (int j = 0; j < src[0].length; j++) {
                dst[i][j] = src[i][j];
            }
        }
    }
}
