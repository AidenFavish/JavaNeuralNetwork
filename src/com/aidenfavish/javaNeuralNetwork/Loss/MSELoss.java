package com.aidenfavish.javaNeuralNetwork.Loss;

import com.aidenfavish.javaNeuralNetwork.Resources.Matrix2D;

public class MSELoss extends Loss {
    public Matrix2D dInputs;

    @Override
    public float[] forward(Matrix2D yPredict, int[] yTrue) {
        return new float[0];
    }

    @Override
    public float[] forward(Matrix2D yPredict, Matrix2D yTrue) {
        return yTrue.plus(yPredict.multiplyConstant(-1)).pow(2).mean(1);

    }

    public void backward(Matrix2D dvalues, Matrix2D yTrue) {
        float samples = dvalues.getMatrix().length;
        float output = dvalues.getMatrix()[0].length;

        dInputs = yTrue.plus(dvalues.multiplyConstant(-1)).multiplyConstant(-2/output).multiplyConstant(1/samples);
    }
}
