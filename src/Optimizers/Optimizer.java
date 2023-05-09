import org.json.simple.JSONObject;

public interface Optimizer {
    void preUpdateParams();
    void updateParams(LayerDense layer);
    void postUpdateParams();
    float getCurrentLearningRate();
    float getLearningRate();
    JSONObject getJSON();
}
