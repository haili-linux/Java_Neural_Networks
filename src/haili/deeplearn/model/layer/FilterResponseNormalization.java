package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class FilterResponseNormalization extends Layer{
    public float[][] w, b;
    public int channel = 1;

    public FilterResponseNormalization(int input_dimension){
        this.id = 8;
        this.input_width = input_dimension;
        this.input_height = 1;
        this.output_width = input_dimension;
        this.output_height = 1;
        this.input_dimension = input_dimension;
        this.output_dimension = input_dimension;

        this.w = new float[channel][input_dimension];
        this.b = new float[channel][input_dimension];
        Arrays.fill(w[0], 1.0f);
    }

    public FilterResponseNormalization(int input_width, int input_height, int input_Dimension){
        this.id = 8;
        init(input_width, input_height, input_Dimension);
    }

    public FilterResponseNormalization(){
        this.id = 8;
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

        this.w = new float[channel][channel_dimension];
        this.b = new float[channel][channel_dimension];
        for(int i = 0 ; i < channel; i++)
            Arrays.fill(w[i], 1.0f);
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

        return new Object[]{outputs, v, v2, x_};
    }


    @Override
    public float[] forward(float[] inputs) {
        return (float[]) forward_list(inputs)[0];
    }



    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        Object[] outputsList = forward_list(inputs);
        float[] v_list = (float[]) outputsList[1];
        float[] v2_list = (float[]) outputsList[2];
        float[][] x_list =  (float[][]) outputsList[3];

        int m = input_width * input_height;
        float[] w_b_deltas = new float[input_dimension * 2];
        float[] last_deltas = new float[input_dimension];

        for(int ci = 0; ci < channel; ci++) {
            int d_c = ci * m;

            float v = v_list[ci];
            float v2 = v2_list[ci];
            float[] x_ = x_list[ci];

            float var0 = (float) Math.pow(v2, -1.5);

            float[] DL_dX_ = new float[m];
            float DL_Dv2 = 0;
            for (int i = 0; i < m; i++) {
                int index = d_c + i;
                DL_dX_[i] = deltas[index] * w[ci][i];
                DL_Dv2 += DL_dX_[i] * inputs[index];
            }
            DL_Dv2 *= -var0 / m;

            for (int i = 0; i < m; i++) {
                int index = d_c + i;
                int index2 = d_c * 2 + i;
                w_b_deltas[index2] = deltas[index] * x_[i];          //dw  检验通过
                w_b_deltas[index2 + m] = deltas[index];

                last_deltas[index] = DL_dX_[i] / v + DL_Dv2 * inputs[index];
            }
        }

        return new float[][]{last_deltas, w_b_deltas};
    }


    @Override
    public void upgradeWeight(float[] weightDeltas) {
        int m = input_width * input_height;
        for (int ci = 0; ci < channel; ci++) {
            int dc = m * ci * 2;
            for (int i = 0; i < m * 2; i++) {
                int index = dc + i;
                if (i < m)
                    w[ci][i] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
                else
                    b[ci][i - m] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
            }
        }
    }

    @Override
    public int getWeightNumber() {
        return input_dimension * 2;
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

        for(float[] wi : w)
            pw.println(SaveData.sFloatArrays("w", wi));
        for(float[] bi : b)
            pw.println(SaveData.sFloatArrays("bias", bi));
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

        w = new float[channel][];
        for(int i = 0; i < channel; i++)
            w[i] = SaveData.getsFloatArrays(in.readLine());

        b = new float[channel][];
        for(int i = 0; i < channel; i++)
            b[i] = SaveData.getsFloatArrays(in.readLine());
    }
}
