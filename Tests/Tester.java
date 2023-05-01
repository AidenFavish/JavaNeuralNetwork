public class Tester {
    public static void main(String[] args) {
        ActivationELU activation1 = new ActivationELU();
        ActivationReLU activation2 = new ActivationReLU();
        Matrix2D test = new Matrix2D(new float[][]{{-1, -2, -3}, {0, 1, 2}});
        activation1.forward(test);
        activation1.backward(activation1.getOutput());
        activation2.forward(test);
        activation2.backward(activation2.getOutput());
        System.out.println("ReLU: \n" + test + "\n\n" + activation2.getOutput() + "\n\n" + activation2.getDInputs());
        System.out.println("\n\nELU: \n" + test + "\n\n" + activation1.getOutput() + "\n\n" + activation1.getDInputs());
    }
}