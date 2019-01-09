package org.gmdh.neuro.fuzzy.utils;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;


@RunWith(MockitoJUnitRunner.class)
public class MathUtilsTest {

    double alpha = 0.25;
    double sigma = 0.5;

    @Test
    public void testGaussFunctionCalculation() {
        double x1 = 1.25;
        double x2 = 2;
        double actual = MathUtils.gauss(x1, alpha, sigma);
    }
}
