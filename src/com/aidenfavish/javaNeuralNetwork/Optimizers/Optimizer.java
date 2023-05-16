package com.aidenfavish.javaNeuralNetwork.Optimizers;

import com.aidenfavish.javaNeuralNetwork.Layers.*;
import com.aidenfavish.javaNeuralNetwork.Resources.*;
import com.aidenfavish.javaNeuralNetwork.ActivationFunctions.*;
import com.aidenfavish.javaNeuralNetwork.Optimizers.*;
import com.aidenfavish.javaNeuralNetwork.Loss.*;

import org.json.simple.JSONObject;

public interface Optimizer {
    void preUpdateParams();
    void updateParams(LayerDense layer);
    void postUpdateParams();
    float getCurrentLearningRate();
    float getLearningRate();
    JSONObject getJSON();
}
