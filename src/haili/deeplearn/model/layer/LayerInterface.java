package haili.deeplearn.model.layer;

public interface LayerInterface {

    void init(int input_width, int input_height, int input_Dimension);

    float[] forward(float[] inputs);
    float[][] backward(float[] inputs, float[] output, float[] deltas);

    void upgradeWeight(float[] WeightDeltas);
}
