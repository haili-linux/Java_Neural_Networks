package haili.deeplearn.model.layer;

public class Layer implements LayerInterface{

    public float learn_rate = 1e-4f;

    @Override
    public float[] forward(float[] inputs) {
        return new float[0];
    }


    @Override
    public float[] backward(float[] inputs, float[] output, float[] deltas) {
        return new float[0];
    }
}
