package haili.deeplearn.model.layer;

import haili.deeplearn.Neuron;
import haili.deeplearn.function.Fuction;
import haili.deeplearn.function.LRelu;

import java.util.Arrays;

public class Conv2D extends Layer{


    int kernel_width, kernel_height, step;

    public float[] w;
    float bias = 0;

    private final int[] index_start;

    Fuction Act_Function;

    public Conv2D(int input_width, int input_height, int kernel_width, int kernel_height, int step, Fuction activation){
        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.step = step;

        this.Act_Function = activation;
        w = new Neuron(kernel_width * kernel_height).w;

        index_start =  new int[w.length];
        for(int i = 0; i < index_start.length; i++){
            int ih = i / kernel_width;
            int iw = i % kernel_width;
            index_start[i] = ih * input_width + iw;
        }
        init(input_width, input_height, input_height*input_width);
    }

    public Conv2D(int kernel_width, int kernel_height, int step, Fuction activation) {
        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.step = step;

        this.Act_Function = activation;
        w = new Neuron(kernel_width * kernel_height).w;

        index_start =  new int[w.length];
        for(int i = 0; i < index_start.length; i++){
            int ih = i / kernel_width;
            int iw = i % kernel_width;
            index_start[i] = ih * input_width + iw;
        }
    }

    @Override
    public void init(int input_width, int input_height, int input_Dimension){
        this.input_width = input_width;
        this.input_height = input_height;

        //超出就忽略
        output_width = (input_width - kernel_width) / step + 1;
        output_height = (input_height - kernel_height) / step + 1;
        input_dimension = input_Dimension;
        output_dimension = output_width * output_height;
    }

    /**
     * forward
     * @param inputs = { X01, X02, ..., X0w, X10, X11, X12, ..., X1w, ...., Xhw}, h = input_height, w = input_width
     * @return outs
     */
    @Override
    public float[] forward(float[] inputs) {
        float[] outputs = new float[output_dimension];
        int[] k_index = index_start.clone();

        for(int ih = 0; ih < output_height; ih++){
            for(int iw = 0; iw < output_width; iw++) {
                int index =  ih * output_width + iw;

                for (int j = 0; j < w.length; j++) {
                    outputs[index] += inputs[k_index[j]] * w[j];
                    if(iw == output_width - 1 )
                        k_index[j] += kernel_width;
                    else
                        k_index[j] += step;
                }
                outputs[index] = Act_Function.f(outputs[index]);
            }
        }
        return outputs;
    }

    @Override
    public float[] backward(float[] inputs, float[] outputs, float[] deltas) {

        float[] last_layer_deltas = new float[inputs.length];
        float[] w_delta = new float[w.length];
        float b_delta = 0;

        int[] k_index = index_start.clone();

        for(int ih = 0; ih < output_height; ih++){
            for(int iw = 0; iw < output_width; iw++) {

                int index =  ih * output_width + iw;

                deltas[index] *= Act_Function.f_derivative(outputs[index]);
                b_delta += deltas[index];

                for (int j = 0; j < w.length; j++) {
                    float delta = deltas[index] * inputs[k_index[j]];
                    w_delta[j] += delta;

                    last_layer_deltas[k_index[j]] += deltas[index] * w[j];

                    if(iw == output_width - 1 )
                        k_index[j] += kernel_width;
                    else
                        k_index[j] += step;
                }

            }
        }

        for (int i = 0; i < w.length; i++) {
            w[i] -= learn_rate * w_delta[i];
        }
        bias -= learn_rate * b_delta;


        return last_layer_deltas;
    }

    @Override
    public String toString() {
        return "Conv2D{" +
                "kernel_width=" + kernel_width +
                ", kernel_height=" + kernel_height +
                ", step=" + step +
                ", Act_Function=" + Act_Function +
                ", input_dimension=" + input_dimension +
                ", input_width=" + input_width +
                ", input_height=" + input_height +
                ", output_dimension=" + output_dimension +
                ", output_width=" + output_width +
                ", output_height=" + output_height +
                '}';
    }
}
