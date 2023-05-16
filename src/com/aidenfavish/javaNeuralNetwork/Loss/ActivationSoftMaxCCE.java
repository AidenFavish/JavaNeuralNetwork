package com.aidenfavish.javaNeuralNetwork.Loss;

import com.aidenfavish.javaNeuralNetwork.Layers.*;
import com.aidenfavish.javaNeuralNetwork.Resources.*;
import com.aidenfavish.javaNeuralNetwork.ActivationFunctions.*;
import org.json.simple.JSONObject;

public class ActivationSoftMaxCCE implements LayerPass
{
    private final ActivationSoftMax activation;
    public CategoricalCrossEntropyLoss loss;
    
    private Matrix2D output;
    
    private Matrix2D dinputs;
    
    public ActivationSoftMaxCCE() {
        activation = new ActivationSoftMax();
        loss = new CategoricalCrossEntropyLoss();
    }
    
    public float forward(Matrix2D inputs, int[] yTrue) {
        activation.forward(inputs);
        
        output = activation.getOutput();
        
        return loss.calculate(output, yTrue);
    }

    public void forward(Matrix2D inputs) {
        activation.forward(inputs);

        output = activation.getOutput();
    }

    @Override
    public void backward(Matrix2D dvalues) {
        // finish method
    }

    public float calculate(int[] yTrue) {
        return loss.calculate(output, yTrue);
    }
    
    public void backward(Matrix2D dvalues, int[] yTrue) {
        int samples = dvalues.getMatrix().length;
        
        dinputs = dvalues.copy();
        
        for (int i = 0; i < samples; i++) {
            dinputs.getMatrix()[i][yTrue[i]] -= 1;
        }
        
        dinputs = dinputs.multiplyConstant(1f/samples);
    }
    
    public Matrix2D getOutput() {
        return output;
    }
    
    public Matrix2D getDInputs() {
        return dinputs;
    }

    @SuppressWarnings("unchecked")
    public JSONObject getJSON() {
        JSONObject ans = new JSONObject();

        ans.put("Name", "com.aidenfavish.javaNeuralNetwork.Loss.ActivationSoftMaxCCE");

        return ans;
    }

    @Override
    public String toString() {
        return activation.toString();
    }
}
