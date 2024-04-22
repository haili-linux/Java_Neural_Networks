package haili.deeplearn.model.layer;

import haili.deeplearn.function.Function;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;

public class ActivationLayer extends Layer{

    public ActivationLayer(Function activation){
        this.id = 9;
        this.activation_function = activation;
    }

    public ActivationLayer(int input_dimension, Function activation){
        this.id = 9;
        this.activation_function = activation;
        init(input_dimension, 1, input_dimension);
    }

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_Dimension;

        this.output_width = input_width;
        this.output_height = input_height;
        this.output_dimension = input_Dimension;
    }

    @Override
    public float[] forward(float[] inputs) {
        float[] outputs = new float[output_dimension];
        for(int i = 0; i < output_dimension; i++)
            outputs[i] = activation_function.f(inputs[i]);

        return outputs;
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[] last_deltas = new float[input_dimension];
        for(int i = 0; i < output_dimension; i++)
            last_deltas[i] = deltas[i] * activation_function.f_derivative(output[i]);

        return  new float[][]{last_deltas, new float[0]};
    }


    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("input_dimension", input_dimension));
        pw.println(SaveData.sInt("input_width", input_width));
        pw.println(SaveData.sInt("input_height", input_height));

        pw.println(SaveData.sInt("output_dimension", output_dimension));
        pw.println(SaveData.sInt("output_width", output_width));
        pw.println(SaveData.sInt("output_height", output_height));

        pw.println(SaveData.sInt("activation", activation_function.id));



    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

        int activation_id = SaveData.getSInt(in.readLine());
        activation_function = Function.getFunctionById(activation_id);
    }

}
