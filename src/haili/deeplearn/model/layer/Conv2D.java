package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizer;
import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.Neuron;
import haili.deeplearn.function.Fuction;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;


public class Conv2D extends Layer{


    int kernel_width, kernel_height, step;

    public float[] w;
    float bias = 0;

    public int[] startConvIndex;

    Fuction Act_Function;

    public Conv2D(int input_width, int input_height, int kernel_width, int kernel_height, int step, Fuction activation){
        id = 2;

        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.step = step;

        this.Act_Function = activation;
        w = new Neuron(kernel_width * kernel_height).w;

        initStartConvIndex();
        init(input_width, input_height, input_height*input_width);
    }

    public Conv2D(int kernel_width, int kernel_height, int step, Fuction activation) {
        id = 2;

        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.step = step;

        this.Act_Function = activation;
        w = new Neuron(kernel_width * kernel_height).w;

        initStartConvIndex();
    }

    private void initStartConvIndex(){
        startConvIndex =  new int[w.length];
        for(int i = 0; i < startConvIndex.length; i++){
            int ih = i / kernel_width;
            int iw = i % kernel_width;
            startConvIndex[i] = ih * input_width + iw;
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

        initStartConvIndex();
    }

    /**
     * forward
     * @param inputs = { X01, X02, ..., X0w, X10, X11, X12, ..., X1w, ...., Xhw}, h = input_height, w = input_width
     * @return outs
     */
    @Override
    public float[] forward(float[] inputs) {
        float[] outputs = new float[output_dimension];
        int[] k_index = startConvIndex.clone();

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
    public float[][] backward(float[] inputs, float[] outputs, float[] deltas) {

        float[] last_layer_deltas = new float[inputs.length];
        float[] w_delta = new float[getWeightNumber()];

        int[] k_index = startConvIndex.clone();

        for(int ih = 0; ih < output_height; ih++){
            for(int iw = 0; iw < output_width; iw++) {

                int index =  ih * output_width + iw;

                deltas[index] *= Act_Function.f_derivative(outputs[index]);
                w_delta[w.length] += deltas[index];

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

        //for (int i = 0; i < w.length; i++) w[i] -= learn_rate * w_delta[i];
        //bias -= learn_rate * w_delta[w.length];

        return new float[][]{ last_layer_deltas,  w_delta};
    }


    @Override
    public void upgradeWeight(float[] weightDeltas) {
        for (int i = 0; i < w.length; i++)
            w[i] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[i], i);

        bias -= learn_rate * deltaOptimizer.DELTA(weightDeltas[w.length], w.length);
    }

    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        deltaOptimizer = deltaOptimizer.getNewObject();
        deltaOptimizer.init(getWeightNumber());
        super.setDeltaOptimizer(deltaOptimizer);
    }

    @Override
    public int getWeightNumber() {
        return w.length + 1;
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

        pw.println(SaveData.sInt("Act_Function_ID", Act_Function.id));
        pw.println(SaveData.sFloat("bias", bias));
        pw.println(SaveData.sFloatArrays("w", w));
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

        Act_Function = Fuction.getFunctionById( SaveData.getSInt(in.readLine()) );
        bias = SaveData.getSFloat(in.readLine());
        w = SaveData.getsFloatArrays(in.readLine());

        initStartConvIndex();
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
                ", w=" + Arrays.toString(w) +
                ", bias=" + bias +
                '}';
    }
}
