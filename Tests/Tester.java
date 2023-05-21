import com.aidenfavish.javaNeuralNetwork.ActivationFunctions.ActivationELU;
import com.aidenfavish.javaNeuralNetwork.ActivationFunctions.ActivationReLU;
import com.aidenfavish.javaNeuralNetwork.Layers.LayerDense;
import com.aidenfavish.javaNeuralNetwork.Loss.ActivationSoftMaxCCE;
import com.aidenfavish.javaNeuralNetwork.Loss.BinaryCrossEntropyLoss;
import com.aidenfavish.javaNeuralNetwork.Loss.MSELoss;
import com.aidenfavish.javaNeuralNetwork.Models.Model;
import com.aidenfavish.javaNeuralNetwork.Optimizers.AdamOptimizer;
import com.aidenfavish.javaNeuralNetwork.Resources.Matrix2D;

import java.sql.SQLOutput;

public class Tester {
    public static void main(String[] args) {
        Model model = new Model("TesterModel", new AdamOptimizer(0.01f, (float)(5 * Math.pow(10, -5)), (float)(1 * Math.pow(10, -7)), 0.9f, 0.999f));
        model.addLayer(new LayerDense(2, 8));
        model.addLayer(new ActivationReLU());
        model.addLayer(new LayerDense(8, 4));
        model.addLayer(new ActivationELU());
        model.addLayer(new LayerDense(4, 3));
        model.addLayer(new ActivationSoftMaxCCE());

        model.save("Models/TesterModel.json");

        model = new Model("Models/TesterModel.json");
        model.train(500, new float[][]{{0.2f, 0.2f}, {0.1f, 0.1f}, {100f, 0.2f}}, new int[]{1, 2, 0});
        System.out.println(model.predict(new float[]{0.15001f, 0.15001f}));
        System.out.println(Model.sample(model.getProb(new float[]{0.15001f, 0.15001f})));

        float[][] testMatrix = new float[][]{{1, 2, 3}, {4, 5, 6}};
        Matrix2D mat = new Matrix2D(testMatrix);
        float[][] testMatrix1 = new float[][]{{1, 4, 5}, {4, 1, 2}};
        Matrix2D mat1 = new Matrix2D(testMatrix1);
        MSELoss testActive = new MSELoss();
        for (float f: testActive.forward(mat, mat1)) {
            System.out.print(f + "\t");
        }
        System.out.println();
        testActive.backward(mat, mat1);
        System.out.println(testActive.dInputs);

    }
}