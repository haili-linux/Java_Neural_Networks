package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.Tanh;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;


public class Conv2DTranspose extends Layer{
    int kernel_width, kernel_height, step;

    int filters;
    int channels;

    public float[][][] w;
    public float[] bias;


    public Conv2DTranspose(int kernel_width, int kernel_height, int filters, int step, Function activation){
        this.id = 6;
        this.kernel_height = kernel_height;
        this.kernel_width = kernel_width;
        this.step = step;
        this.filters = filters;
        this.activation_function = activation;
    }

    public Conv2DTranspose(int kernel_width, int kernel_height, int filters, int step, Function activation, boolean use_bias){
        this.id = 6;
        this.kernel_height = kernel_height;
        this.kernel_width = kernel_width;
        this.step = step;
        this.filters = filters;
        this.activation_function = activation;
        this.use_bias = use_bias;
    }

    public Conv2DTranspose(int input_width, int input_height, int kernel_width, int kernel_height, int filters, int channels, int step, Function activation){
        this.id = 6;
        this.kernel_height = kernel_height;
        this.kernel_width = kernel_width;
        this.step = step;
        this.filters = filters;
        input_dimension = input_width * input_height * channels;
        this.activation_function = activation;
        init(input_width, input_height, input_dimension);
    }

    public Conv2DTranspose(int input_width, int input_height, int kernel_width, int kernel_height, int filters, int channels, int step, Function activation, boolean use_bias){
        this.id = 6;
        this.kernel_height = kernel_height;
        this.kernel_width = kernel_width;
        this.step = step;
        this.filters = filters;
        input_dimension = input_width * input_height * channels;
        this.activation_function = activation;
        init(input_width, input_height, input_dimension);
        this.use_bias = use_bias;
    }



    @Override
    public void init(int input_width, int input_height, int input_dimension) {
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_dimension;

        if(input_dimension % (input_width * input_height) != 0){
            System.out.println(this.getClass() + "  Error: input_dimension % (input_width * input_height) != 0" );
            System.exit(0);
        }

        channels = input_dimension / (input_width * input_height);
        w = new float[filters][channels][];

        for(int i = 0; i < filters; i++)
            for(int j = 0; j < channels; j++) {
                w[i][j] = GaussRandomArrays(kernel_width * kernel_height);
                for (int k = 0; k < w[i][j].length; k++){
                    w[i][j][k] /= channels;
                }
            }

        output_width = step * input_width + kernel_width - step ;
        output_height = step * input_height + kernel_height - step;
        this.output_dimension = output_width * output_height * filters;

        bias = new float[filters];

        wn = filters * channels * w[0][0].length + filters;
        startDeConvIndex = startDeConvIndexStart();
    }

    private int[] startDeConvIndex;

    /**
     * x, y，坐标系为计算机屏幕坐标，左上角为原点
     * @return 输出该点映射到输出数组上的对应的点的索引值
     */
    private int[] startDeConvIndexStart(){
        //x = 0, y = 0,时
        int[] array = new int[kernel_width * kernel_height];
        for(int layer_y = 0; layer_y < kernel_height; layer_y++) {
            for(int x = 0; x < kernel_width; x++){
                int index = layer_y * kernel_width + x;
                array[index] = layer_y * output_width + x;
            }
        }
        return array;
    }

    /**
     * forward
     * @param inputs = { channel1 (X01, X02, ..., X0w, X10, X11, X12, ..., X1w, ...., Xhw), channel2 (X01,...) ...., }, h = input_height, w = input_width
     * @return outs [ channel1 (y01, y02,..., y11, y12,....yhw), channel2, ...., channelN ],
     */
    @Override
    public float[] forward(float[] inputs) {
        float[] output = new float[this.output_dimension];
        int outDimension_filter = output_width * output_height;
        int input_channel_dimension = input_width * input_height;
        for(int filtes_i = 0; filtes_i < filters; filtes_i++){
            int out_filter_d = outDimension_filter * filtes_i;

            int[] convIndex = startDeConvIndex.clone();
            for(int input_point = 0; input_point < input_channel_dimension; input_point++){
                int x_point = input_point % input_width;

                for(int channel_index = 0; channel_index < channels; channel_index++){
                    for (int i = 0; i < convIndex.length; i++) {
                        int inputs_d_channel_index = channel_index * input_channel_dimension;
                        output[convIndex[i] + out_filter_d] += w[filtes_i][channel_index][i] * inputs[input_point + inputs_d_channel_index];
                    }
                }

                if(x_point == input_width - 1){
                    for (int i = 0; i < convIndex.length; i++)
                        convIndex[i] += (step - 1) * output_width + kernel_width;
                } else {
                    for (int i = 0; i < convIndex.length; i++)
                        convIndex[i] += step;
                }
            }
        }

        for(int i = 0; i < output.length; i++){
            int index_filter = i / (outDimension_filter);
            output[i] += bias[index_filter];
            output[i] = activation_function.f(output[i]);
        }

        return output;
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        int outDimension_filter = output_width * output_height;
        int var0 =  channels * w[0][0].length;
        float[] w_deltas = new float[getWeightNumber()];

        int bias_index_d = filters * var0;
        for(int i = 0; i < output.length; i++){
            deltas[i] *= activation_function.f_derivative(output[i]);
            if(use_bias) {
                int index_filter = i / (outDimension_filter);
                w_deltas[bias_index_d + index_filter] += deltas[i];
            }
        }

        float[] inputs_deltas = new float[input_dimension];

        int input_channel_dimension = input_width * input_height;
        for(int filter_i = 0; filter_i < filters; filter_i++){
            int out_filter_d = outDimension_filter * filter_i;
            int w_deltas_filter_d = filter_i * var0;

            int[] convIndex = startDeConvIndex.clone();
            for(int input_point = 0; input_point < input_channel_dimension; input_point++){
                int x_point = input_point % input_width;

                for(int channel_index = 0; channel_index < channels; channel_index++){
                    for (int i = 0; i < convIndex.length; i++) {
                        int output_index = convIndex[i] + out_filter_d;
                        int w_deltas_index = w_deltas_filter_d + channel_index * convIndex.length + i;
                        int inputs_d_channel_index = channel_index * input_channel_dimension + input_point;
                        w_deltas[w_deltas_index] += deltas[output_index] * inputs[ inputs_d_channel_index];
                        inputs_deltas[inputs_d_channel_index] += deltas[output_index] * w[filter_i][channel_index][i];
                    }
                }

                if(x_point == input_width - 1){
                    for (int i = 0; i < convIndex.length; i++)
                        convIndex[i] += (step - 1) * output_width + kernel_width;
                } else {
                    for (int i = 0; i < convIndex.length; i++)
                        convIndex[i] += step;
                }
            }
        }
        return new float[][]{inputs_deltas, w_deltas};
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) {
        int bias_index_d = filters * channels * w[0][0].length;

        //update w
        for (int i = 0; i < bias_index_d; i++){
            int index_filter = i /  (channels * w[0][0].length);
            int index_channel =  (i % (channels * w[0][0].length)) /  w[0][0].length;
            int index_w = (i % (channels * w[0][0].length)) %  w[0][0].length;
            w[index_filter][index_channel][index_w] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[i], i);
        }

        //update bias
        for(int i = bias_index_d; i < weightDeltas.length; i++){
            bias[i - bias_index_d] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[i], i);
        }
    }

    private int wn;
    @Override
    public int getWeightNumber() {
        return wn;
    }

    @Override
    public int getWeightNumber_Train() {
        if(use_bias)
            return getWeightNumber();
        else
            return getWeightNumber() - filters;
    }

    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        deltaOptimizer = deltaOptimizer.getNewObject();
        deltaOptimizer.init(getWeightNumber());
        super.setDeltaOptimizer(deltaOptimizer);
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
        pw.println(SaveData.sInt("step", step));
        pw.println(SaveData.sInt("channel", channels));
        pw.println(SaveData.sInt("filters", filters));

        int ub = 1;
        if(!use_bias) ub = 0;
        pw.println(SaveData.sInt("use_bias", ub));

        pw.println(SaveData.sInt("Act_Function_ID", activation_function.id));

        pw.println(SaveData.sFloatArrays("bias", bias));

        for(int i = 0; i < filters; i++) {
            for (int j = 0; j < channels; j++)
                pw.println(SaveData.sFloatArrays("w[" + i + "][" + j + "]", w[i][j]));
        }

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
        step = SaveData.getSInt(in.readLine());
        channels = SaveData.getSInt(in.readLine());
        filters = SaveData.getSInt(in.readLine());

        int ub =  SaveData.getSInt(in.readLine());
        if(ub == 0)
            this.use_bias = false;

        activation_function = Function.getFunctionById( SaveData.getSInt(in.readLine()) );

        bias = SaveData.getsFloatArrays(in.readLine());

        w = new float[filters][channels][];
        for(int i = 0; i < filters; i++) {
            for (int j = 0; j < channels; j++)
                w[i][j] = SaveData.getsFloatArrays(in.readLine());
        }

        wn = filters * channels * w[0][0].length + filters;
        startDeConvIndex = startDeConvIndexStart();
    }

}
