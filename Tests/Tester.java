public class Tester {
    public static void main(String[] args) {
        Model model = new Model("TesterModel", new AdamOptimizer(0.01f, (float)(5 * Math.pow(10, -5)), (float)(1 * Math.pow(10, -7)), 0.9f, 0.999f));
        model.addLayer(new LayerDense(2, 16));
        model.addLayer(new ActivationReLU());
        model.addLayer(new LayerDense(16, 3));
        model.addLayer(new ActivationSoftMaxCCE());

        model.save("Models/TesterModel.json");
    }
}