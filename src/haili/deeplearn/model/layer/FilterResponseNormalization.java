package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class FilterResponseNormalization extends Layer{
    public float[] w, b;
    public int channel = 1;


    public FilterResponseNormalization(int input_width, int input_height, int input_Dimension){
        this.id = 8;
        init(input_width, input_height, input_Dimension);
    }

    public FilterResponseNormalization(int input_width, int input_height, int input_Dimension, Function activation){
        this.id = 8;
        this.activation_function = activation;
        init(input_width, input_height, input_Dimension);
    }

    public FilterResponseNormalization(int input_dimension){
        this.id = 8;
        init(input_dimension, 1, input_dimension);
    }

    public FilterResponseNormalization(int input_dimension, Function activation){
        this.id = 8;
        this.activation_function = activation;
        init(input_dimension, 1, input_dimension);
    }

    public FilterResponseNormalization(){
        this.id = 8;
    }

    public FilterResponseNormalization(Function activation){
        this.id = 8;
        this.activation_function = activation;
    }


    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        this.input_width = input_width;
        this.input_height = input_height;

        this.output_width = input_width;
        this.output_height = input_height;

        this.input_dimension = input_Dimension;
        this.output_dimension = input_Dimension;

        int channel_dimension = input_width * input_height;
        this.channel = input_Dimension / channel_dimension;

        int v0 = channel > 1 ? channel:input_Dimension;

        this.w = new float[v0];
        this.b = new float[v0];
        Arrays.fill(w, 1.0f);
    }

    public Object[] forward_list(float[] inputs){
        Object[] output = null;
        if(saveHiddenLayerOutput && hiddenLayerOutputMap.containsKey(inputs)) {
            output = (Object[]) hiddenLayerOutputMap.get(inputs);
            if(output != null)
                return output;
        }

        //计算
        output = forward_List(inputs);

        // 保存中间输出
        if(saveHiddenLayerOutput){
            hiddenLayerOutputMap.put(inputs, output);
        }
        return output;
    }

    // output max: input_dimension
    private Object[] forward_List(float[] inputs) {
        float[] outputs = new float[output_dimension];

        int m = input_dimension;

        float v2 = 0;
        for(float xi: inputs)
            v2 += xi * xi;

        v2 = v2 / m + 1e-6f;
        float v = (float) Math.sqrt(v2);

        int dc = w.length == input_dimension ? 1 : input_width * input_height;
        float[] x_ = new float[m];
        for(int ci = 0; ci < w.length; ci++){

            // channel[] * w[i] + b[i]
            // 当channel.len (dc)== 1, c * w[i] + b[i]
            int start = ci * dc;
            int end = start + dc;
            for(int i = start; i < end; i++){
                x_[i] = inputs[i] / v;
                outputs[i] = activation_function.f(w[ci] * x_[i] + b[ci]);
            }
        }

        /*
        int dimension_channel = input_width * input_height;
        float[] v2 = new float[channel];
        float[] v = new float[channel];
        float[][] x_ = new float[channel][];

        for(int ci = 0; ci < channel; ci++) {
            int d_c = ci * dimension_channel;

            for (int i = 0; i < dimension_channel; i++) {
                int index = d_c + i;
                v2[ci] += inputs[index] * inputs[index];
            }
            v2[ci] = v2[ci] / dimension_channel + 1e-6f;

            v[ci] = (float) Math.sqrt(v2[ci]);

            x_[ci] = new float[dimension_channel];
            for (int i = 0; i < dimension_channel; i++) {
                int index = d_c + i;
                x_[ci][i] = inputs[index] / v[ci];
                outputs[index] = w[ci][i] * x_[ci][i] + b[ci][i];
            }
        }
        */

        return new Object[]{outputs, v, v2, x_};
    }


    @Override
    public float[] forward(float[] inputs) {
        return (float[]) forward_list(inputs)[0];
    }



    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        Object[] outputsList = forward_list(inputs);
        float v = (float)outputsList[1];
        float v2 = (float)outputsList[2];
        float[] x_ = (float[])outputsList[3];

        float[] w_b_deltas = new float[w.length * 2];
        float[] last_deltas = new float[inputs.length];



        float var0 = (float) Math.pow(v2, -1.5);

        int m = input_width * input_height;



        float[] DL_dX_ = new float[inputs.length];
        float DL_Dv2 = 0;

        int dc = w.length == input_dimension ? 1 : input_width * input_height;
        for (int ci = 0; ci < w.length; ci++) {

            int start = ci * dc;
            int end = start + dc;
            for(int i = start; i < end; i++){
                deltas[i] *= activation_function.f_derivative(output[i]);
                DL_dX_[i] = deltas[i] * w[ci];
                DL_Dv2 += DL_dX_[i] * inputs[i];
            }
        }

        DL_Dv2 *= -var0 / inputs.length;

        for (int ci = 0; ci < w.length; ci++) {
            int start = ci * dc;
            int end = start + dc;
            for(int i = start; i < end; i++){
                w_b_deltas[ci] += deltas[i] * x_[i];          //dw  检验通过
                w_b_deltas[ci + w.length] += deltas[i];

                last_deltas[i] = DL_dX_[i] / v + DL_Dv2 * inputs[i];
            }
        }

        return new float[][]{last_deltas, w_b_deltas};
    }



    @Override
    public void upgradeWeight(float[] weightDeltas) {
        int m = w.length;
        for (int i = 0; i < m * 2; i++) {
            if (i < m)
                w[i] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[i], i);
            else
                b[i - m] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[i], i);
        }
    }



    @Override
    public int getWeightNumber() {
        return w.length + b.length;
    }

    @Override
    public int getWeightNumber_Train() {
        return getWeightNumber();
    }

    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        deltaOptimizer = deltaOptimizer.getNewObject();
        deltaOptimizer.init(getWeightNumber() );
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

        pw.println(SaveData.sInt("activation", activation_function.id));

        pw.println(SaveData.sFloatArrays("w", w));
        pw.println(SaveData.sFloatArrays("bias", b));
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

        channel = input_dimension / (input_width * input_height);

        int act_id = SaveData.getSInt(in.readLine());
        activation_function = Function.getFunctionById(act_id);

        w = SaveData.getsFloatArrays(in.readLine());
        b = SaveData.getsFloatArrays(in.readLine());
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder= new StringBuilder();
        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        char[] c0 = new char[32 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "(" + output_width + ", " + output_height + ", " + output_dimension + ")";
        int v0 = 25 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');
        int param = getWeightNumber();

        stringBuilder.append(name).append(c0).append(output_shape).append(c1).append(param);
        return stringBuilder.toString();
    }
}
