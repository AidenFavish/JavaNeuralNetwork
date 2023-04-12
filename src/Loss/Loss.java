
public abstract class Loss
{
    
    public float calculate(Matrix2D output, int[] y) {
        float[] sampleLosses = this.forward(output, y);
        float sum = 0;
        for (float x: sampleLosses) {
            sum += x;
        }
        
        return sum / sampleLosses.length;
    }
    
    public abstract float[] forward(Matrix2D yPredict, int[] yTrue);
    
    public static Matrix2D clip(Matrix2D x, float lower, float upper) {
        float[][] ans = new float[x.getMatrix().length][x.getMatrix()[0].length];
        float temp;
        for (int r = 0; r < x.getMatrix().length; r++) {
            for (int c = 0; c < x.getMatrix()[0].length; c++) {
                temp = x.getMatrix()[r][c];
                if (temp < lower)
                    temp = lower;
                if (temp > upper)
                    temp = upper;
                ans[r][c] = temp;
            }
        }
        return new Matrix2D(ans);
    }
}
