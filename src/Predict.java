
public class Predict
{
    private float[][] dataX;
    private int[] dataY;
    private float[][] validX;
    private int[] validY;
    private int sideLength;

    public Predict(float[][] dataX, int[] dataY, int percentValid, int sideLength) { // percent valid is counter intuitive fyi
        this.sideLength = sideLength;
        this.dataX = new float[dataX.length * percentValid / 100][2];
        this.dataY = new int[dataY.length * percentValid / 100];
        this.validX = new float[dataX.length - dataX.length * percentValid / 100][2];
        this.validY = new int[dataY.length - dataY.length * percentValid / 100];
        shuffle(dataX, dataY);
        for (int i = 0; i < dataX.length * percentValid / 100; i++) {
            this.dataX[i] = dataX[i];
            this.dataY[i] = dataY[i];
        }
        for (int i = dataX.length * percentValid / 100 + 1; i < dataX.length; i++) {
            this.validX[i - (dataX.length * percentValid / 100 + 1)] = dataX[i];
            this.validY[i - (dataX.length * percentValid / 100 + 1)] = dataY[i];
        }
        System.out.println("Data sorted");
    }

    public int[][] predict() {
        // Train the model
        float[] regulars = new float[]{0, (float)(Math.pow(10, -4) * 5), 0, (float)(Math.pow(10, -4) * 5)};
        LayerDense dense1 = new LayerDense(2, 64, regulars);
        ActivationReLU activation1 = new ActivationReLU();
        LayerDense dense2 = new LayerDense(64, 16);
        ActivationReLU activation2 = new ActivationReLU();
        LayerDense dense3 = new LayerDense(16, 3);
        ActivationSoftMaxCCE lossActivation = new ActivationSoftMaxCCE();
        AdamOptimizer optimizer = new AdamOptimizer(0.01f, (float)(5 * Math.pow(10, -5)), (float)(1 * Math.pow(10, -7)), 0.9f, 0.999f);
        float dataLoss;
        float regLoss;
        float loss;
        int[] predictions;
        int ctr;
        float accuracy;

        for (int epoch = 0; epoch < 10001; epoch++) {
            dense1.forward(new Matrix2D(dataX));
            activation1.forward(dense1.getOutput());
            dense2.forward(activation1.getOutput());
            activation2.forward(dense2.getOutput());
            dense3.forward(activation2.getOutput());

            dataLoss = lossActivation.forward(dense3.getOutput(), dataY);
            regLoss = lossActivation.loss.regularization_loss(dense1) + lossActivation.loss.regularization_loss(dense2) + lossActivation.loss.regularization_loss(dense3);
            loss = dataLoss + regLoss;

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
            dense3.backward(lossActivation.getDInputs());
            activation2.backward(dense3.getDInputs());
            dense2.backward(activation2.getDInputs());
            activation1.backward(dense2.getDInputs());
            dense1.backward(activation1.getDInputs());

            optimizer.preUpdateParams();
            optimizer.updateParams(dense1);
            optimizer.updateParams(dense2);
            optimizer.updateParams(dense3);
            optimizer.postUpdateParams();
        }
        
        // Validate the model
        dense1.forward(new Matrix2D(validX));
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());
        activation2.forward(dense2.getOutput());
        dense3.forward(activation2.getOutput());
        loss = lossActivation.forward(dense3.getOutput(), validY);
        predictions = lossActivation.getOutput().argmax(1);
        ctr = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == validY[i])
                ctr++;
        }
        accuracy = (float) ctr / predictions.length;
        System.out.println("validation, acc: " + accuracy + ", loss: " + loss);
        
        // predict!
        float[][] map = new float[sideLength * sideLength][2];
        int ctr1 = 0;
        for (int i = 0; i < sideLength; i++) {
            for (int j = 0; j < sideLength; j++) {
                map[ctr1++] = new float[]{i, j};
            }
        }
        dense1.forward(new Matrix2D(map));
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());
        activation2.forward(dense2.getOutput());
        dense3.forward(activation2.getOutput());
        loss = lossActivation.forward(dense3.getOutput(), new int[sideLength * sideLength]);
        predictions = lossActivation.getOutput().argmax(1);
        
        int[][] results = new int[sideLength][sideLength];
        ctr1 = 0;
        for (int r = 0; r < sideLength; r++) {
            for (int c = 0; c < sideLength; c++) {
                results[r][c] = predictions[ctr1++];
            }
        }
        System.out.println("Prediction complete");
        
        return results;
    }

    private void shuffle(float[][] x, int[] y) {
        int temp;
        float[] temp1;
        int temp2;
        for (int i = 0; i < x.length; i++) {
            temp = (int)(Math.random() * x.length);
            temp1 = x[temp];
            temp2 = y[temp];
            x[temp] = x[i];
            y[temp] = y[i];
            x[i] = temp1;
            y[i] = temp2;
        }
    }
}
