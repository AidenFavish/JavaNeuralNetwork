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
        model.train(500, new float[][]{{0.2f, 0.2f}, {0.1f, 0.1f}}, new int[]{1, 2});
        System.out.println(model.predict(new float[]{0.15001f, 0.15001f}));
    }
}