package haili.deeplearn.model.layer;

import haili.deeplearn.Neuron;
import haili.deeplearn.function.Fuction;
import haili.deeplearn.function.LRelu;

import java.util.Arrays;

public class Conv2D extends Layer{


    int kernel_width, kernel_height, step;

    int input_width, input_height;
    int output_width, output_height;

    public float[] w;
    float bias = 0;

    private final int[] index_start;

    Fuction Act_Function;

    public Conv2D(int input_width, int input_height, int kernel_width, int kernel_height, int step, Fuction activation){
        this.input_width = input_width;
        this.input_height = input_height;
        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.step = step;

        //超出就忽略
        output_width = (input_width - kernel_width) / step + 1;
        output_height = (input_height - kernel_height) / step + 1;

        Act_Function = activation;
        w = new Neuron(kernel_width * kernel_height).w;

        index_start =  new int[w.length];
        for(int i = 0; i < index_start.length; i++){
            int ih = i / kernel_width;
            int iw = i % kernel_width;
            index_start[i] = ih * input_width + iw;
        }
        System.out.println("out dimension: w" + output_width  + "   h:" + output_height);
    }

    /**
     * forward
     * @param inputs = { X01, X02, ..., X0w, X10, X11, X12, ..., X1w, ...., Xhw}, h = input_height, w = input_width
     * @return out
     */
    @Override
    public float[] forward(float[] inputs) {
        float[] outputs = new float[output_width * output_height];
        int[] k_index = index_start.clone();

        for(int ih = 0; ih < output_height; ih++){
            for(int iw = 0; iw < output_width; iw++) {
                //System.out.println(Arrays.toString(k_index));
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
                //System.out.print("   dy = deltas[" + index + "] = " + deltas[index] );
                b_delta += deltas[index];

                for (int j = 0; j < w.length; j++) {
                    float delta = deltas[index] * inputs[k_index[j]];
                    //System.out.println("   deltas[" + index + "] = " + deltas[index] +   "  inputs[k_index[j] = " + inputs[k_index[j]] + "   " + delta);


                    //int ix = k_index[j] % 3, iy = k_index[j] /3;
                    //int x = j % 2 , y = j / 2 ;

                    w_delta[j] += delta;
                    //System.out.println(w_delta[j]);

                    last_layer_deltas[k_index[j]] += deltas[index] * w[j];
                    //outputs[index] =+ inputs[k_index[j]] * w[j];

                    if(iw == output_width - 1 )
                        k_index[j] += kernel_width;
                    else
                        k_index[j] += step;
                }
                //System.out.println("\n");
                //outputs[index] = Act_Function.f(outputs[index]);
            }
        }

//        System.out.println(" Delta = " + Arrays.toString(deltas));
//        System.out.println(" W_delta = " + Arrays.toString(w_delta));
//        System.out.println(" b_delta = " + b_delta);
//        System.out.println(" input_delta = " + Arrays.toString(last_layer_deltas));
        for (int i = 0; i < w.length; i++) {
            w[i] -= learn_rate * w_delta[i];
        }
        bias -= learn_rate * b_delta;

//        System.out.println("===========================");
//        for(int i = 0; i < last_layer_deltas.length; i ++){
//            if(i!=0 && i%input_width ==0) System.out.print("\n");
//            System.out.print(last_layer_deltas[i] + "  " );
//        }
//        System.out.println("\n===========================");

        return last_layer_deltas;
    }
}
