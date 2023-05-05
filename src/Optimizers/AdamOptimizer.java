import org.json.simple.JSONObject;

public class AdamOptimizer implements Optimizer
{
    private float learningRate;
    private float currentLearningRate;
    private float decay;
    private int iterations;
    private float epsilon;
    private float beta1;
    private float beta2;
    
    public AdamOptimizer(float learningRate, float decay, float epsilon, float beta1, float beta2) {
        this.learningRate = learningRate;
        this.currentLearningRate = learningRate;
        this.decay = decay;
        this.iterations = 0;
        this.epsilon = epsilon;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }
    
    public void preUpdateParams() {
        currentLearningRate = learningRate * (1f/(1f + decay * iterations));
    }
    
    public void updateParams(LayerDense layer) {
        if (layer.weightCache == null) {
            layer.weightMomentums = Matrix2D.zerosLike(layer.getWeights());
            layer.weightCache = Matrix2D.zerosLike(layer.getWeights());
            layer.biasMomentums = Matrix2D.zerosLike(layer.getBiases());
            layer.biasCache = Matrix2D.zerosLike(layer.getBiases());
        }
        
        /*Matrix2D weightUpdates = layer.weightMomentums.multiplyConstant(0.9f).plus(layer.getDWeights().multiplyConstant(-1 * currentLearningRate));
        layer.weightMomentums = weightUpdates;
        layer.setWeights(layer.getWeights().plus(weightUpdates));
        
        Matrix2D biasUpdates = layer.biasMomentums.multiplyConstant(0.9f).plus(layer.getDBiases().multiplyConstant(-1 * currentLearningRate));
        layer.biasMomentums = biasUpdates;
        layer.setBiases(layer.getBiases().plus(biasUpdates)); // SGD */
        
        layer.weightMomentums = layer.weightMomentums.multiplyConstant(beta1).plus(layer.getDWeights().multiplyConstant(1 - beta1));
        layer.biasMomentums = layer.biasMomentums.multiplyConstant(beta1).plus(layer.getDBiases().multiplyConstant(1 - beta1));
        
        Matrix2D weightMomentumsCorrected = layer.weightMomentums.multiplyConstant(1f/(1 - (float)Math.pow(beta1, iterations + 1)));
        Matrix2D biasMomentumsCorrected = layer.biasMomentums.multiplyConstant(1f/(1 - (float)Math.pow(beta1, iterations + 1)));
        
        layer.weightCache = layer.weightCache.multiplyConstant(beta2).plus(layer.getDWeights().pow(2f).multiplyConstant(1 - beta2));
        layer.biasCache = layer.biasCache.multiplyConstant(beta2).plus(layer.getDBiases().pow(2f).multiplyConstant(1 - beta2));
        
        Matrix2D weightCacheCorrected = layer.weightCache.multiplyConstant(1f/(1 - (float)Math.pow(beta2, iterations + 1)));
        Matrix2D biasCacheCorrected = layer.biasCache.multiplyConstant(1f/(1 - (float)Math.pow(beta2, iterations + 1)));
        
        layer.setWeights(layer.getWeights().plus(weightMomentumsCorrected.multiplyConstant(-1 * currentLearningRate).divideBy(weightCacheCorrected.pow(0.5f).plus(epsilon))));
        layer.setBiases(layer.getBiases().plus(biasMomentumsCorrected.multiplyConstant(-1 * currentLearningRate).divideBy(biasCacheCorrected.pow(0.5f).plus(epsilon))));
    
        
    }
    
    public void postUpdateParams() {
        iterations += 1;
    }
    
    public float getCurrentLearningRate() {
        return currentLearningRate;
    }

    public float getLearningRate() { return learningRate; }

    @SuppressWarnings("unchecked")
    public JSONObject getJSON() {
        JSONObject ans = new JSONObject();
        ans.put("Learning Rate", learningRate);
        ans.put("Decay", decay);
        ans.put("Epsilon", epsilon);
        ans.put("Beta1", beta1);
        ans.put("Beta2", beta2);
        return ans;
    }

}
