package haili.deeplearn.model.layer;

import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class SplitLayer extends Layer{

    int start_index;

    public SplitLayer(int start_index, int length){
        this.id = 15;
        this.output_width = length;
        this.output_height = 1;
        this.output_dimension = length;

        this.start_index = start_index;
    }

    @Override
    public void init(int input_width, int input_height, int input_dimension) {
        //this.input_width = input_width;
        //this.input_height = input_height;
        //this.input_dimension = input_dimension;
    }

    @Override
    public float[] forward(float[] inputs) {
        float[] outputs = new float[output_dimension];
        System.arraycopy(inputs, start_index, outputs, 0, output_dimension);
        //for(int i = 0; i < output_dimension; i++)
        //    outputs[i] = inputs[start_index + i];
        return outputs;
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[] inputs_deltas = new float[inputs.length];

        System.arraycopy(deltas, 0, inputs_deltas, start_index, output_dimension);
        //for(int i = 0; i < output_dimension; i++)
        //    inputs_deltas[start_index + i] = deltas[i];

        return new float[][]{inputs_deltas, new float[0]};
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("output_width", output_width));
        pw.println(SaveData.sInt("output_height", output_height));
        pw.println(SaveData.sInt("output_dimension", output_dimension));

        pw.println(SaveData.sInt("start_index", start_index));
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());
        output_dimension = SaveData.getSInt(in.readLine());

        start_index = SaveData.getSInt(in.readLine());
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder= new StringBuilder();
        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        char[] c0 = new char[32 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "(start:" + start_index +  ", len:" + output_dimension  + ")";

        int v0 = 25 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');
        int param = getWeightNumber_Train();

        char[] c2 = new char[0];
        if(deepOfSequential > 0) {
            c2 = new char[deepOfSequential * 2];
            Arrays.fill(c2, ' ');
        }

        stringBuilder.append(c2).append(name).append(c0).append(output_shape).append(c1).append(param);

        return stringBuilder.toString();
    }
}
