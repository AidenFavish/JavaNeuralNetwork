
public class ActivationSoftMax
{
    private Matrix2D inputs;
    private Matrix2D output;
    
    private Matrix2D dinputs;
    
    public void forward(Matrix2D inputs) {
        this.inputs = inputs;
        
        Matrix2D expValues = inputs.max(1).multiplyConstant(-1f).plus(inputs).exp();
        
        Matrix2D probabilities = expValues.divideBy(expValues.sum(1));
        
        this.output = probabilities;
    }
    
    public void backward(Matrix2D dvalues) {
        dinputs = Matrix2D.zeros(dvalues.getMatrix().length, dvalues.getMatrix()[0].length);
        
        int index = 0;
        
        //More to come
    }
    
    public Matrix2D getOutput() {
        return output;
    }
    
    public Matrix2D getDInputs() {
        return dinputs;
    }
}
