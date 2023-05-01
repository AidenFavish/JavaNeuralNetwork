public class ActivationELU {

    private float alpha;
    private Matrix2D inputs;
    private Matrix2D output;

    private Matrix2D dinputs;

    public ActivationELU() {
        this.alpha = 1f;
    }
    public ActivationELU(float alpha) {
        this.alpha = alpha;
    }

    public void forward(Matrix2D inputs) {
        this.inputs = inputs;
        Matrix2D copyInput = inputs.copy();
        this.output = copyInput.conditionOperate(0, x -> (float)Math.exp(x) - 1).multiplyConstant(alpha);
    }

    public void backward(Matrix2D dvalues) {
        Matrix2D temp = this.inputs.copy();
        temp.conditionModify2(temp, 0, 1);
        this.dinputs = temp.conditionOperate(0, x -> (float)Math.exp(x)).multiplyConstant(alpha).times(dvalues);

    }

    public Matrix2D predictions(Matrix2D outputs) {
        return outputs;
    }

    public Matrix2D getOutput() {
        return output;
    }

    public Matrix2D getDInputs() {
        return dinputs;
    }

    public float getAlpha() { return alpha; }

    public void setAlpha(float x) { alpha = x; }
}
