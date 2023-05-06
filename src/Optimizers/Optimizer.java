import java.io.FileWriter;
import java.io.IOException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import java.util.*;
public interface Optimizer {
    void preUpdateParams();
    void updateParams(LayerDense layer);
    void postUpdateParams();
    float getCurrentLearningRate();
    float getLearningRate();
    JSONObject getJSON();
}
