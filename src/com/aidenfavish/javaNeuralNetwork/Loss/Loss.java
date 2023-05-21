package com.aidenfavish.javaNeuralNetwork.Loss;

import com.aidenfavish.javaNeuralNetwork.Layers.*;
import com.aidenfavish.javaNeuralNetwork.Resources.*;

public abstract class Loss
{
    public float regularization_loss(LayerDense layer) {

        float regularizationLoss = 0;


        if (layer.weightRegularizerL1 > 0) {
            regularizationLoss += layer.getWeights().abs().sum() * layer.weightRegularizerL1;
        }

        if (layer.weightRegularizerL2 > 0) {
            regularizationLoss += layer.getWeights().pow(2).sum() * layer.weightRegularizerL2;
        }

        if (layer.biasRegularizerL1 > 0) {
            regularizationLoss += layer.getBiases().abs().sum() * layer.biasRegularizerL1;
        }

        if (layer.biasRegularizerL2 > 0) {
            regularizationLoss += layer.getBiases().pow(2).sum() * layer.biasRegularizerL2;
        }
        return regularizationLoss;
    }
    
    public float calculate(Matrix2D output, int[] y) {
        float[] sampleLosses = this.forward(output, y);
        float sum = 0;
        for (float x: sampleLosses) {
            sum += x;
        }
        
        return sum / sampleLosses.length;
    }

    public float calculate(Matrix2D output, Matrix2D y) {
        float[] sampleLosses = this.forward(output, y);
        float sum = 0;
        for (float x: sampleLosses) {
            sum += x;
        }

        return sum / sampleLosses.length;
    }
    
    public abstract float[] forward(Matrix2D yPredict, int[] yTrue);
    public abstract float[] forward(Matrix2D yPredict, Matrix2D yTrue);
    
    public static Matrix2D clip(Matrix2D x, float lower, float upper) {
        float[][] ans = new float[x.getMatrix().length][x.getMatrix()[0].length];
        float temp;
        for (int r = 0; r < x.getMatrix().length; r++) {
            for (int c = 0; c < x.getMatrix()[0].length; c++) {
                temp = x.getMatrix()[r][c];
                if (temp < lower)
                    temp = lower;
                if (temp > upper)
                    temp = upper;
                ans[r][c] = temp;
            }
        }
        return new Matrix2D(ans);
    }
}
