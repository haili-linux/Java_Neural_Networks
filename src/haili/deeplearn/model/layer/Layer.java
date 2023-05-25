package haili.deeplearn.model.layer;

public class Layer implements LayerInterface{

    public float learn_rate = 1e-4f;

    public int input_dimension = 0;
    public int input_width = 0, input_height = 0;

    public int output_dimension = 0;
    public int output_width = 0, output_height = 0;


    @Override
    public void init(int input_width, int input_height, int input_Dimension) { }

    @Override
    public float[] forward(float[] inputs) {
        return new float[0];
    }


    @Override
    public float[] backward(float[] inputs, float[] output, float[] deltas) {
        return new float[0];
    }
}
