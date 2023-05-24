package haili.deeplearn.model.layer;

public interface LayerInterface {
    float[] forward(float[] inputs);
    float[] backward(float[] inputs, float[] output, float[] deltas);
}
