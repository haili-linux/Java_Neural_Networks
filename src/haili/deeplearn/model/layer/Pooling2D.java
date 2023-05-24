package haili.deeplearn.model.layer;

import java.util.Arrays;

public class Pooling2D extends Layer{

    
    int kernel_width, kernel_height;

    int input_width, input_height;
    int output_width, output_height;


    //padding = 0;
    public Pooling2D(int input_width, int input_height, int kernel_width, int kernel_height){
        this.input_width = input_width;
        this.input_height = input_height;
        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;

        output_width = input_width /kernel_width;
        if(input_width % kernel_width > 0) output_width++;

        output_height = input_height / kernel_height;
        if(input_height % kernel_height > 0) output_height++;

    }

    @Override
    public float[] forward(float[] inputs) {
        float[] out = new float[output_width * output_height];

        for (int ih = 0; ih < input_height; ih += kernel_height)
            for (int iw = 0; iw < input_width; iw += kernel_width) {

                int oh = ih / kernel_height, ow = iw / kernel_width;
                int index_o = oh * output_width + ow;
                out[index_o] = -99999999999.0f;

                for (int i = ih; i < ih + kernel_height; i++) {
                    for (int j = iw; j < iw + kernel_width; j++) {
                        if (i < input_height && j < input_width) {
                            int index = i * input_width + j;
                            if (inputs[index] > out[index_o])
                                out[index_o] = inputs[index];
                        } else {
                            if (0 > out[index_o]) out[index_o] = 0;
                        }
                    }
                }
            }

        return out;
    }

    @Override
    public float[] backward(float[] inputs, float[] output, float[] deltas) {
        float[] last_layer_deltas = new float[input_width * input_height];

        for (int ih = 0; ih < output_height; ih ++)
            for (int iw = 0; iw < output_width; iw ++) {
                int index = ih * output_width + iw;
                float delta = deltas[index] / kernel_height / kernel_width;
                for (int i = ih * kernel_height; i < (ih + 1) * kernel_height && i < input_height; i++)
                    for (int j = iw * kernel_width; j < (iw + 1) * kernel_width && j < input_width; j++) {
                        last_layer_deltas[i*input_width + j] = delta;
                    }

            }
        return last_layer_deltas;
    }
}
