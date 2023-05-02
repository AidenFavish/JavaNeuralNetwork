public interface Optimizer {
    public void preUpdateParams();
    public void updateParams(LayerDense layer);
    public void postUpdateParams();
    public float getCurrentLearningRate();
    public float getLearningRate();
}
