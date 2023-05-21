package com.aidenfavish.javaNeuralNetwork.Models;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import com.aidenfavish.javaNeuralNetwork.Layers.*;
import com.aidenfavish.javaNeuralNetwork.Resources.*;
import com.aidenfavish.javaNeuralNetwork.ActivationFunctions.*;
import com.aidenfavish.javaNeuralNetwork.Optimizers.*;
import com.aidenfavish.javaNeuralNetwork.Loss.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.util.*;

/**
 * This class is designed to hold layers and optimization methods to abstract the complex training and inference
 * processes behind the scenes.
 */
@SuppressWarnings("unused")
public class Model {
    private String name;
    private Optimizer optimizer;
    private List<LayerPass> network;

    public Model(String name, Optimizer optimizer) {
        this.name = name;
        this.optimizer = optimizer;
        this.network = new ArrayList<>();
    }

    public Model(String path) {
        try {
            JSONParser parser = new JSONParser();
            JSONObject obj = (JSONObject) parser.parse(new FileReader(path));

            this.name = (String) obj.get("Name");
            this.optimizer = decodeOptimizer((JSONObject) obj.get("com.aidenfavish.javaNeuralNetwork.Optimizers.Optimizer"));
            this.network = decodeNetwork((JSONArray) obj.get("Network"));

        } catch(Exception e) {
            System.out.println("Error thrown loading model: " + e);
        }
    }

    public void addLayer(LayerPass layer) {
        network.add(layer);
    }

    public void addLayer(int index, LayerPass layer) {
        network.add(index, layer);
    }

    public List<LayerPass> getNetwork() { return network; }

    public void train(int epochs, float[][] dataX, int[] dataY) { // For CCEL
        ActivationSoftMaxCCE lossActivation;
        float dataLoss;
        float regLoss;
        float loss;
        int[] predictions;
        int ctr;
        float accuracy;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            pass(dataX);
            lossActivation = (ActivationSoftMaxCCE) network.get(network.size() - 1);
            dataLoss = lossActivation.calculate(dataY);
            regLoss = 0;
            for (LayerPass layerPass : network) {
                if (layerPass instanceof LayerDense) {
                    regLoss += lossActivation.loss.regularization_loss((LayerDense) layerPass);
                }
            }
            loss = regLoss + dataLoss;

            predictions = lossActivation.getOutput().argmax(1);
            ctr = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (predictions[i] == dataY[i])
                    ctr++;
            }
            accuracy = (float) ctr / predictions.length;

            if (epoch % 100 == 0) {
                System.out.println("epoch: " + epoch + ", acc: " + accuracy + ", loss: " + loss + ", dataLoss: " + dataLoss + ", regLoss: " + regLoss + ", lr: " + optimizer.getCurrentLearningRate());
            }

            lossActivation.backward(lossActivation.getOutput(), dataY);
            for (int i = network.size() - 2; i >= 0; i--) {
                network.get(i).backward(network.get(i + 1).getDInputs());
            }

            optimizer.preUpdateParams();
            for (LayerPass layer: network) {
                if (layer instanceof LayerDense) {
                    optimizer.updateParams((LayerDense) layer);
                }
            }
            optimizer.postUpdateParams();
        }
    }

    public float validate(float[][] dataX, int[] dataY) {
        ActivationSoftMaxCCE lossActivation;
        float dataLoss;
        float regLoss;
        float loss;
        int[] predictions;
        int ctr;
        float accuracy;
        pass(dataX);
        lossActivation = (ActivationSoftMaxCCE) network.get(network.size() - 1);
        dataLoss = lossActivation.calculate(dataY);
        regLoss = 0;
        for (LayerPass layerPass : network) {
            if (layerPass instanceof LayerDense) {
                regLoss += lossActivation.loss.regularization_loss((LayerDense) layerPass);
            }
        }
        loss = regLoss + dataLoss;

        predictions = lossActivation.getOutput().argmax(1);
        ctr = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == dataY[i])
                ctr++;
        }
        accuracy = (float) ctr / predictions.length;
        System.out.println("Validate: acc: " + accuracy + ", loss: " + loss + ", dataLoss: " + dataLoss + ", regLoss: " + regLoss);

        return loss;
    }

    public int predict(float[] data) {
        pass(new float[][]{data});
        int[] predictions = network.get(network.size() - 1).getOutput().argmax(1);
        return predictions[0];
    }

    public float predictDisct(float[] data) {
        pass(new float[][]{data});
        Matrix2D predictions = network.get(network.size() - 2).getOutput();
        return predictions.getMatrix()[0][0];
    }

    public float[] getProb(float[] data) {
        pass(new float[][]{data});
        return network.get(network.size() - 1).getOutput().getMatrix()[0];
    }

    public float[] getProb(float[][] data) {
        pass(data);
        return network.get(network.size() - 1).getOutput().mean(0);
    }

    public void pass(float[][] dataX) {
        network.get(0).forward(new Matrix2D(dataX));
        for (int layer = 1; layer < network.size(); layer++) {
            network.get(layer).forward(network.get(layer-1).getOutput());
        }
    }

    @SuppressWarnings("unchecked")
    public void save(String path) {
        JSONObject model = new JSONObject();

        //Insert the data
        model.put("Name", name);
        model.put("com.aidenfavish.javaNeuralNetwork.Optimizers.Optimizer", optimizer.getJSON());
        model.put("Network", constructJSONNetwork());

        try {
            FileWriter file = new FileWriter(path);
            file.write(model.toJSONString());
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
            //
        }
        System.out.println("JSON file created: "+model);
    }

    @SuppressWarnings("unchecked")
    private JSONArray constructJSONNetwork() {
        JSONArray ans = new JSONArray();

        for (LayerPass layer: network) {
            ans.add(layer.getJSON());
        }

        return ans;
    }

    private Optimizer decodeOptimizer(JSONObject obj) {
        Optimizer ans;
        String objName = (String) obj.get("Name");
        if (objName.equals("Adam com.aidenfavish.javaNeuralNetwork.Optimizers.Optimizer")) {
            ans = new AdamOptimizer((float)((double) obj.get("Learning Rate")), (float)((double) obj.get("Decay")), (float)((double) obj.get("Epsilon")), (float)((double) obj.get("Beta1")), (float)((double) obj.get("Beta2")));
        } else {
            ans = new AdamOptimizer(0.01f, (float)(5 * Math.pow(10, -5)), (float)(1 * Math.pow(10, -7)), 0.9f, 0.999f);
        }
        return ans;
    }

    private List<LayerPass> decodeNetwork(JSONArray arr) {
        List<LayerPass> ans = new ArrayList<>();

        for (Object obj: arr) {
            ans.add(decodeLayer((JSONObject) obj));
        }

        return ans;
    }

    private LayerPass decodeLayer(JSONObject obj) {
        String layerName = (String) obj.get("Name");
        LayerPass ans;

        switch (layerName) {
            case "com.aidenfavish.javaNeuralNetwork.Layers.LayerDense" ->
                    ans = new LayerDense(decodeMatrix((JSONArray) obj.get("Weights")), decodeMatrix((JSONArray) obj.get("Biases")), (float)((double) obj.get("WeightRegular1")), (float)((double) obj.get("WeightRegular2")), (float)((double) obj.get("BiasRegular1")), (float)((double) obj.get("BiasRegular2")));
            case "com.aidenfavish.javaNeuralNetwork.ActivationFunctions.ActivationELU" -> ans = new ActivationELU();
            case "com.aidenfavish.javaNeuralNetwork.ActivationFunctions.ActivationReLU" -> ans = new ActivationReLU();
            case "com.aidenfavish.javaNeuralNetwork.ActivationFunctions.ActivationSoftMax" -> ans = new ActivationSoftMax();
            case "com.aidenfavish.javaNeuralNetwork.Loss.ActivationSoftMaxCCE" -> ans = new ActivationSoftMaxCCE();
            default -> {
                System.out.println("Layer Pass corrupted");
                ans = null;
            }
        }

        return ans;
    }

    private Matrix2D decodeMatrix(JSONArray arr) {
        float[][] ans = new float[arr.size()][((JSONArray) arr.get(0)).size()];

        for (int r = 0; r < ans.length; r++) {
            for (int c = 0; c < ans[0].length; c++) {
                ans[r][c] = (float) ((double) ((JSONArray) arr.get(r)).get(c));
            }
        }

        return new Matrix2D(ans);
    }

    @Override
    public String toString() {
        StringBuilder ans = new StringBuilder("--------------------------\nName: " + name + "\ncom.aidenfavish.javaNeuralNetwork.Optimizers.Optimizer: " + optimizer + "\nNetwork:\n");
        for (LayerPass layer: network) {
            ans.append("\t").append(layer).append("\n");
        }
        return ans + "--------------------------";
    }

    public static int sample(float[] prob) {
        float r = (float)Math.random();
        float temp = 0;
        for (int i = 0; i < prob.length; i++) {
            temp += prob[i];
            if (r < temp) {
                return i;
            }
        }
        return 0;
    }


}