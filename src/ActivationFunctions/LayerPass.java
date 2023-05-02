public interface LayerPass {
    public void forward(Matrix2D inputs);

    public void backward(Matrix2D dvalues);

    public Matrix2D getOutput();

    public Matrix2D getDInputs();

}
