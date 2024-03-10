package haili.deeplearn.model.layer.softmax;

import haili.deeplearn.model.layer.Layer;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;

public class SoftmaxLayer extends Layer {

    public SoftmaxLayer(int input_dimension){
        id = 4;
        this.input_dimension = input_dimension;
        this.output_dimension = input_dimension;
    }

    @Override
    public float[] forward(float[] inputs) {
        float total = 0;
        for(int i = 0; i < inputs.length; i++){
            inputs[i] = (float) Math.exp(inputs[i]);
            total += inputs[i];
        }

        for(int i = 0; i < inputs.length; i++)
            inputs[i] /= total;

        return inputs;
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[] t_delta = new float[deltas.length];
        for (int i = 0; i < deltas.length; i++){
            for(int j = 0; j < deltas.length; j++)
                if(j == i)
                    t_delta[j] += deltas[i] * output[i] * ( 1 - output[i] );
                else
                    t_delta[j] += deltas[i] * (-output[i]) * output[j];
        }

        return new float[][]{t_delta, new float[0]};
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

    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {

        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

    }

}
