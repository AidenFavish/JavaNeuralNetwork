package com.aidenfavish.javaNeuralNetwork.Resources;

public interface Environment {
    public int getObsSpaceShape();
    public int getActSpaceShape();
    public float[] reset();
    public void setDone(boolean x);
    public void render();
    public Object[] step(int action);
}
