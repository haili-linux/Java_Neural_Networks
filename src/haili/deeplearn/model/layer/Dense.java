package haili.deeplearn.model.layer;

import haili.deeplearn.Neuron;
import haili.deeplearn.function.Fuction;

public class Dense extends Layer{
    public final int input_Dimension;
    public final int output_Dimension;

    public Neuron[] neurons;

    public Dense(int input_Dimension, int output_Dimension, Fuction activation){
        this.input_Dimension = input_Dimension;
        this.output_Dimension = output_Dimension;

        neurons = new Neuron[output_Dimension];
        //w = new float[output_Dimension][];
        //bias = new float[output_Dimension];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(input_Dimension, activation);
            //w[i] = neurons[i].w;
            //bias[i] = neurons[i].b;
        }
    }

    @Override
    public float[] forward(float[] inputs) {
        float[] output = new float[output_Dimension];
        for (int i = 0; i < output.length; i++)
            output[i] = neurons[i].out_notSave(inputs);

        return output;
    }


    /**
     * 反向传播
     * @param inputs 本层输入
     * @param output 本层输出
     * @param deltas 下一层传回的梯度，对应本层每个神经元
     * @return 传给下一层的梯度，对应下一层每个神经元
     */
    @Override
    public float[] backward(float[] inputs, float[] output, float[] deltas) {

        float[] last_layer_deltas = new float[input_Dimension];

        float[][] deltas_this_w = new float[output_Dimension][input_Dimension];
        for(int i = 0; i < neurons.length; i++){

            deltas[i] *= neurons[i].ACT_function.f_derivative(output[i]);
            neurons[i].b -= learn_rate * deltas[i];

            for(int j = 0; j < neurons[i].w.length; j++) {
                last_layer_deltas[i] += deltas[i] * neurons[i].w[j];

                deltas_this_w[i][j] = deltas[i] * inputs[j];
                neurons[i].setW(j, neurons[i].w[j] - learn_rate *  deltas_this_w[i][j]);
            }

        }

        return last_layer_deltas;
    }
}
