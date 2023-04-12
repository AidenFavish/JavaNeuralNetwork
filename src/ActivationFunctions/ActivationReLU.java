
public class ActivationReLU
{
    private Matrix2D inputs;
    private Matrix2D output;
    
    private Matrix2D dinputs;
    
    public void forward(Matrix2D inputs) {
        this.inputs = inputs;
        Matrix2D copyInput = inputs.copy();
        this.output = copyInput.conditionModify1(copyInput, 0, 0);
    }
    
    public void backward(Matrix2D dvalues) {
        this.dinputs = dvalues.copy();
        
        this.dinputs.conditionModify1(inputs, 0, 0);
        
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
}
