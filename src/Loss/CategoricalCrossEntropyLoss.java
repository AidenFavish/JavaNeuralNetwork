
public class CategoricalCrossEntropyLoss extends Loss
{
    public float[] forward(Matrix2D yPredict, int[] yTrue) {
        int samples = yPredict.getMatrix().length;
        
        Matrix2D yPredictClipped = Loss.clip(yPredict, (float)Math.pow(10, -7), (float)(1-Math.pow(10, -7)));
        
        float[] correctConfidences = new float[samples];
        
        for (int i = 0; i < samples; i++) {
            correctConfidences[i] = -1 * (float)Math.log(yPredictClipped.getMatrix()[i][yTrue[i]]);
        }
        
        return correctConfidences;
    }
    
}
