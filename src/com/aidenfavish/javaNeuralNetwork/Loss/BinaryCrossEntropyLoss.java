package com.aidenfavish.javaNeuralNetwork.Loss;

import com.aidenfavish.javaNeuralNetwork.Resources.Matrix2D;

public class BinaryCrossEntropyLoss extends Loss {

    public Matrix2D dInputs;

    @Override
    public float[] forward(Matrix2D yPredict, int[] yTrue) {
        return null;
    }

    @Override
    public float[] forward(Matrix2D yPredict, Matrix2D yTrue) {

        Matrix2D yPredictClipped = Loss.clip(yPredict, (float)Math.pow(10, -7), (float)(1-Math.pow(10, -7)));

        Matrix2D sampleLosses = yPredictClipped.log().times(yTrue).plus(yTrue.plus(-1f).multiplyConstant(-1).times(yPredictClipped.plus(-1f).multiplyConstant(-1).log()));
        return sampleLosses.multiplyConstant(-1).mean(1);
    }

    public void backward(Matrix2D dvalues, Matrix2D yTrue) {
        int samples = dvalues.getMatrix().length;
        int outputs = dvalues.getMatrix()[0].length;
        Matrix2D clippedDValues = Loss.clip(dvalues, (float)Math.pow(10, -7), (float)(1-Math.pow(10, -7)));

        dInputs = yTrue.divideBy(clippedDValues).plus(yTrue.plus(-1).divideBy(clippedDValues.multiplyConstant(-1).plus(1))).multiplyConstant(-1/(float)outputs);
        dInputs = dInputs.multiplyConstant(1/(float)samples);
        // Large floating point error
    }
}

