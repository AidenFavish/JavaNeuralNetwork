
public class LayerDense
{
    private Matrix2D weights;
    private Matrix2D biases;
    public float weightRegularizerL1;
    public float weightRegularizerL2;
    public float biasRegularizerL1;
    public float biasRegularizerL2;
    
    private Matrix2D inputs;
    private Matrix2D output;
    
    private Matrix2D dinputs;
    private Matrix2D dweights;
    private Matrix2D dbiases;
    
    private Matrix2D dL1;
    
    public Matrix2D weightMomentums;
    public Matrix2D weightCache;
    public Matrix2D biasMomentums;
    public Matrix2D biasCache;
    
    public LayerDense(int n_inputs, int n_neurons) {
        // Args may consists of wr1, wr2, br1, br2
        weights = Matrix2D.random(n_inputs, n_neurons).multiplyConstant(0.01f);
        
        biases = Matrix2D.zeros(1, n_neurons);

        weightRegularizerL1 = 0;
        weightRegularizerL2 = 0;
        biasRegularizerL1 = 0;
        biasRegularizerL2 = 0;
    }

    public LayerDense(int n_inputs, int n_neurons, float[] regulars) {
        // Args may consists of wr1, wr2, br1, br2
        weights = Matrix2D.random(n_inputs, n_neurons).multiplyConstant(0.01f);

        biases = Matrix2D.zeros(1, n_neurons);

        weightRegularizerL1 = regulars[0];
        weightRegularizerL2 = regulars[1];
        biasRegularizerL1 = regulars[2];
        biasRegularizerL2 = regulars[3];
    }
    
    public void forward(Matrix2D inputs) {
        this.inputs = inputs;
        this.output = inputs.dot(weights).plus(biases);
    }
    
    public void backward(Matrix2D dvalues) {
        this.dweights = inputs.T().dot(dvalues);
        this.dbiases = dvalues.sum(0);
        
        if (weightRegularizerL1 > 0) {
            dL1 = Matrix2D.onesLike(weights);
            dL1.conditionModify1(weights, 0, -1);
            dweights = dweights.plus(dL1.multiplyConstant(weightRegularizerL1));
        }
        
        if (weightRegularizerL2 > 0) {
            dweights = dweights.plus(weights.multiplyConstant(weightRegularizerL2 * 2));
        }
        
        if (biasRegularizerL1 > 0) {
            dL1 = Matrix2D.onesLike(biases);
            dL1.conditionModify1(biases, 0, -1);
            dbiases = dbiases.plus(dL1.multiplyConstant(biasRegularizerL1));
        }
        
        if (biasRegularizerL2 > 0) {
            dbiases = dbiases.plus(biases.multiplyConstant(2 * biasRegularizerL2));
        }
        
        dinputs = dvalues.dot(weights.T());
    }
    
    public Matrix2D getWeights() {
        return weights;
    }
    
    public Matrix2D getBiases() {
        return biases;
    }
    
    public Matrix2D getDWeights() {
        return dweights;
    }
    
    public Matrix2D getDBiases() {
        return dbiases;
    }
    
    public Matrix2D getOutput() {
        return output;
    }
    
    public Matrix2D getDInputs() {
        return dinputs;
    }
    
    public void setWeights(Matrix2D x) {
        weights = x;
    }
    
    public void setBiases(Matrix2D x) {
        biases = x;
    }
}
