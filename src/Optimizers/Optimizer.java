import java.io.FileWriter;
import java.io.IOException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import java.util.*;
public interface Optimizer {
    public void preUpdateParams();
    public void updateParams(LayerDense layer);
    public void postUpdateParams();
    public float getCurrentLearningRate();
    public float getLearningRate();
    public JSONObject getJSON();
}
