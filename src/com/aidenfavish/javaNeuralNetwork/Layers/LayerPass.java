package com.aidenfavish.javaNeuralNetwork.Layers;

import com.aidenfavish.javaNeuralNetwork.Resources.*;
import org.json.simple.JSONObject;

public interface LayerPass {
    void forward(Matrix2D inputs);

    void backward(Matrix2D dvalues);

    Matrix2D getOutput();

    Matrix2D getDInputs();

    JSONObject getJSON();

}
