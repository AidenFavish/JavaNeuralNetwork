import java.util.*;

public class Model {
    private String name;
    private Optimizer optimizer;
    private List<LayerPass> network;

    public Model(String name, Optimizer optimizer) {
        this.name = name;
        this.optimizer = optimizer;
        this.network = new ArrayList<LayerPass>();
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
        for (int epoch = 0; epoch < epochs; epoch++) {
            pass(dataX);

            lossActivation = (ActivationSoftMaxCCE) network.get(network.size() - 1);
            dataLoss = lossActivation.calculate(dataY);
            regLoss = 0;
            for (int i = 0; i < network.size(); i++) {
                if (network.get(i) instanceof LayerDense) {
                    regLoss += lossActivation.loss.regularization_loss((LayerDense) network.get(i));
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
                network.get(i).backward(network.get(i - 1).getDInputs());
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

    public void pass(float[][] dataX) {
        network.get(0).forward(new Matrix2D(dataX));
        for (int layer = 1; layer < network.size(); layer++) {
            network.get(layer).forward(network.get(layer-1).getOutput());
        }
    }

}