package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizer;
import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.Neuron;
import haili.deeplearn.function.Fuction;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;


public class Conv2D extends Layer{


    int kernel_width, kernel_height, step;

    int filters;
    int channels;

    public float[][][] w;
    public float[] bias;

    public int[] startConvIndex;

    Fuction Act_Function;

    public Conv2D(int input_width, int input_height, int kernel_width, int kernel_height, int filters, int channels, int step, Fuction activation){
        id = 2;

        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.filters = filters;
        this.channels = channels;
        this.step = step;

        this.Act_Function = activation;

        init(input_width, input_height, input_height * input_width * channels);
    }


    // use in Sequential
    public Conv2D(int kernel_width, int kernel_height, int filters, int step, Fuction activation) {
        id = 2;

        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.filters = filters;
        this.step = step;

        this.Act_Function = activation;

    }


    private void initStartConvIndex(){
        startConvIndex =  new int[w[0][0].length];
        for(int i = 0; i < startConvIndex.length; i++){
            int ih = i / kernel_width;
            int iw = i % kernel_width;
            startConvIndex[i] = ih * input_width + iw;
        }
    }


    @Override
    public void init(int input_width, int input_height, int input_dimension){

        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_dimension;

        if(input_dimension % (input_width * input_height) != 0){
            System.out.println(this.getClass().toString() + "  Error: input_dimension % (input_width * input_height) != 0" );
            return;
        }

        channels = input_dimension / (input_width * input_height);
        w = new float[filters][channels][];
        bias = new float[filters];

        for(int i = 0; i < filters; i++)
            for(int j = 0; j < channels; j++)
                w[i][j] = new Neuron(kernel_width * kernel_height).w;


        //超出就忽略
        output_width = (input_width - kernel_width) / step + 1;
        output_height = (input_height - kernel_height) / step + 1;
        output_dimension = output_width * output_height * filters;

        init2();

        initStartConvIndex();
    }

    private void init2(){
        one_channel_dimension = output_width * output_height;
        one_filter_wn = w[0][0].length * channels + 1;

        wn = (w[0][0].length * channels + 1) * filters;
    }

    /**
     * forward
     * @param inputs = { channel1 (X01, X02, ..., X0w, X10, X11, X12, ..., X1w, ...., Xhw), channel2 (X01,...) ...., }, h = input_height, w = input_width
     * @return outs [channel1 (y01, y02,..., y11, y12,....yhw), channel2, ...., channelN]
     */
    @Override
    public float[] forward(float[] inputs) {

        float[] outputs = new float[output_dimension];

        for(int filters_i = 0; filters_i < filters; filters_i++) {

            int filters_dx = filters_i * one_channel_dimension;

            for (int channels_j = 0; channels_j < channels; channels_j++) {

                int[] k_index = startConvIndex.clone();

                //channel dx
                for (int ik = 0; ik < k_index.length; ik++)
                    k_index[ik] += channels_j * one_channel_dimension;

                int index = 0;

                //one channel
                for (int ih = 0; ih < output_height; ih++) {
                    for (int iw = 0; iw < output_width; iw++) {

                        index = filters_dx + (ih * output_width + iw);


                        for (int j = 0; j < w[filters_i][channels_j].length; j++) {
                            outputs[index] += inputs[k_index[j]] * w[filters_i][channels_j][j];

                            if (iw == output_width - 1)
                                k_index[j] += kernel_width;
                            else
                                k_index[j] += step;
                        }

                    }
                }

                if(channels_j == channels-1)
                    outputs[index] = Act_Function.f( outputs[index] + bias[filters_i] );
            }//end channels
        }//end filters

        return outputs;
    }


    private int one_channel_dimension;
    private int one_filter_wn;

    /**
     * backward
     * @param inputs 本层输入
     * @param outputs 本层forward输出
     * @param deltas 上一层传递到的梯度
     * @return { 下一层梯度, 本层参数梯度 }
     */
    @Override
    public float[][] backward(float[] inputs, float[] outputs, float[] deltas) {

        float[] last_layer_deltas = new float[inputs.length];
		//[filter][Chanel][h*w]
        float[] w_delta = new float[getWeightNumber()];


        for(int filters_i = 0; filters_i < filters; filters_i++) {

            int filters_dx = filters_i * one_channel_dimension;
			int w_index_dx = filters_i * one_filter_wn;

            for (int channels_j = 0; channels_j < channels; channels_j++) {
				
				    int[] k_index = startConvIndex.clone();

                    for (int ik = 0; ik < k_index.length; ik++)
                         k_index[ik] += channels_j * one_channel_dimension;
				
					int w_channel_dx = w_index_dx + channels_j * w[filters_i][channels_j].length;	 
			        int dxchannel = channels * w[0][0].length;
					
                    for (int ih = 0; ih < output_height; ih++) {
                        for (int iw = 0; iw < output_width; iw++) {

                            int index = filters_dx + ih * output_width + iw;

                            deltas[index] *= Act_Function.f_derivative(outputs[index]);
                            w_delta[w_index_dx + dxchannel] += deltas[index];

                            for (int j = 0; j < w[filters_i][channels_j].length; j++) {
                                float delta = deltas[index] * inputs[k_index[j]];
                                w_delta[w_channel_dx + j] += delta;

                                //last_layer_deltas[k_index[j]] += deltas[index] * w[j];

                                if (iw == output_width - 1)
                                    k_index[j] += kernel_width;
                                else
                                    k_index[j] += step;
                            }

                        }
                    }
            }// end channels_j
        }

        return new float[][]{ last_layer_deltas,  w_delta};
    }


    @Override
    public void upgradeWeight(float[] weightDeltas) {
		int index = 0;
        for (int i = 0; i < filters; i++){
			for(int j = 0; j < channels; j++){
				for(int k = 0; k < w[i][j].length; k++){
					w[i][j][k] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
					index++;
				}
			}
			bias[i] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
			index++;
		}
   }

    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        deltaOptimizer = deltaOptimizer.getNewObject();
        deltaOptimizer.init(getWeightNumber());
        super.setDeltaOptimizer(deltaOptimizer);
    }


    private int wn = 0;
    @Override
    public int getWeightNumber() {
        return wn;
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

        pw.println(SaveData.sInt("Act_Function_ID", Act_Function.id));
        pw.println(SaveData.sFloatArrays("bias", bias));
		for(int i = 0; i < filters; i++)
		   for(int j = 0; j < channels; j++){
			   pw.println(SaveData.sFloatArrays("w[" + i +"][" + j + "]", w[i][j]));
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
		

        Act_Function = Fuction.getFunctionById( SaveData.getSInt(in.readLine()) );

        bias = SaveData.getsFloatArrays(in.readLine());

        w = new float[filters][channels][];
		for(int i = 0; i < filters; i++)
			for(int j = 0; j < channels; j++){
				w[i][j] = SaveData.getsFloatArrays(in.readLine());
			}

		init2();
        initStartConvIndex();
    }

    @Override
    public String toString() {
        return "Conv2D{" +
                "kernel_width=" + kernel_width +
                ", kernel_height=" + kernel_height +
                ", step=" + step +
                ", filters=" + filters +
                ", channels=" + channels +
                ", w=" + Arrays.toString(w) +
                ", bias=" + Arrays.toString(bias) +
                ", startConvIndex=" + Arrays.toString(startConvIndex) +
                ", Act_Function=" + Act_Function +
                ", one_channel_dimension=" + one_channel_dimension +
                ", one_filter_wn=" + one_filter_wn +
                ", wn=" + wn +
                ", id=" + id +
                ", learn_rate=" + learn_rate +
                ", input_dimension=" + input_dimension +
                ", input_width=" + input_width +
                ", input_height=" + input_height +
                ", output_dimension=" + output_dimension +
                ", output_width=" + output_width +
                ", output_height=" + output_height +
                '}';
    }
}
