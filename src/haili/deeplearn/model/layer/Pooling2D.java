package haili.deeplearn.model.layer;

import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;

public class Pooling2D extends Layer{

    int kernel_width, kernel_height;
    int channels;

    //padding = 0;
    public Pooling2D(int input_width, int input_height, int kernel_width, int kernel_height){
        id = 3;

        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;

       init(input_width, input_height, 0);

    }

    public Pooling2D(int kernel_width, int kernel_height){
        id = 3;

        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
    }

    @Override
    public void init(int input_width, int input_height, int input_dimension){
        this.input_width = input_width;
        this.input_height = input_height;

        if(input_dimension % (input_width * input_height) != 0){
            System.out.println(this.getClass().toString() + "  Error: input_dimension % (input_width * input_height) != 0" );
            return;
        }

        channels = input_dimension / (input_width * input_height);

        output_width = input_width /kernel_width;
        if(input_width % kernel_width > 0) output_width++;

        output_height = input_height / kernel_height;
        if(input_height % kernel_height > 0) output_height++;

        output_dimension = output_height * output_width * channels;
        this.input_dimension = input_dimension;

        one_channels_outputs = output_height * output_width;
        one_inputs_channels = input_width * input_height;
    }

    private int one_channels_outputs;
    private int one_inputs_channels;

    @Override
    public float[] forward(float[] inputs) {
        float[] out = new float[output_dimension];

        for(int channel_i = 0; channel_i < channels; channel_i++) {

            int output_dx = channel_i * one_channels_outputs;
            int input_dx =  channel_i * one_inputs_channels;

            for (int ih = 0; ih < input_height; ih += kernel_height)
                for (int iw = 0; iw < input_width; iw += kernel_width) {

                    int oh = ih / kernel_height, ow = iw / kernel_width;

                    int index_o = output_dx  +  oh * output_width + ow;
                    out[index_o] = -99999999999.0f;

                    for (int i = ih; i < ih + kernel_height; i++) {
                        for (int j = iw; j < iw + kernel_width; j++) {

                            if (i < input_height && j < input_width) {

                                int index = input_dx + i * input_width + j;

                                if (inputs[index] > out[index_o]) out[index_o] = inputs[index];

                            } else if (0 > out[index_o]) out[index_o] = 0;

                        }
                    }
                }
        }

        return out;
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[] last_layer_deltas = new float[input_dimension];

        for(int channel_i = 0; channel_i < channels; channel_i++) {

            int output_dx = channel_i * one_channels_outputs;
            int input_dx = channel_i * one_inputs_channels;

            for (int ih = 0; ih < output_height; ih++)
                for (int iw = 0; iw < output_width; iw++) {
                    int index = output_dx + ih * output_width + iw;

                    float delta = deltas[index] / kernel_height / kernel_width;

                    for (int i = ih * kernel_height; i < (ih + 1) * kernel_height && i < input_height; i++)
                        for (int j = iw * kernel_width; j < (iw + 1) * kernel_width && j < input_width; j++) {
                            last_layer_deltas[input_dx + i * input_width + j] = delta;
                        }

                }
        }
        return new float[][]{ last_layer_deltas, new float[0] };
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

        pw.println(SaveData.sInt("kernel_width", kernel_width));
        pw.println(SaveData.sInt("kernel_height", kernel_height));
        pw.println(SaveData.sInt("channels", channels));
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

        kernel_width = SaveData.getSInt(in.readLine());
        kernel_height = SaveData.getSInt(in.readLine());
        channels = SaveData.getSInt(in.readLine());

        init(input_width, input_height, input_dimension);
    }


    @Override
    public String toString() {
        return "Pooling2D{" +
                "input_dimension=" + input_dimension +
                ", input_width=" + input_width +
                ", input_height=" + input_height +
                ", output_dimension=" + output_dimension +
                ", output_width=" + output_width +
                ", output_height=" + output_height +
                ", kernel_width=" + kernel_width +
                ", kernel_height=" + kernel_height +
                '}';
    }
}
