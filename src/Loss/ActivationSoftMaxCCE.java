
public class ActivationSoftMaxCCE
{
    private ActivationSoftMax activation;
    private CategoricalCrossEntropyLoss loss;
    
    private Matrix2D output;
    
    private Matrix2D dinputs;
    
    public ActivationSoftMaxCCE() {
        activation = new ActivationSoftMax();
        loss = new CategoricalCrossEntropyLoss();
    }
    
    public float forward(Matrix2D inputs, int[] yTrue) {
        activation.forward(inputs);
        
        output = activation.getOutput();
        
        return loss.calculate(output, yTrue);
    }
    
    public void backward(Matrix2D dvalues, int[] yTrue) {
        int samples = dvalues.getMatrix().length;
        
        dinputs = dvalues.copy();
        
        for (int i = 0; i < samples; i++) {
            dinputs.getMatrix()[i][yTrue[i]] -= 1;
        }
        
        dinputs = dinputs.multiplyConstant(1f/samples);
    }
    
    public Matrix2D getOutput() {
        return output;
    }
    
    public Matrix2D getDInputs() {
        return dinputs;
    }
}
