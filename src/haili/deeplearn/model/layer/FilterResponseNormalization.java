package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class FilterResponseNormalization extends Layer{
    public float[] w, b;

    public FilterResponseNormalization(int input_dimension){
        this.id = 8;
        this.input_width = input_dimension;
        this.input_height = 1;
        this.output_width = input_dimension;
        this.output_height = 1;
        this.input_dimension = input_dimension;
        this.output_dimension = input_dimension;

        this.w = new float[input_dimension];
        this.b = new float[input_dimension];
        Arrays.fill(w, 1.0f);
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

        this.w = new float[input_dimension];
        this.b = new float[input_dimension];
        Arrays.fill(w, 1.0f);
    }

    // output max: input_dimension
    public float[][] forward_list(float[] inputs) {
        float[] outputs = new float[output_dimension];

        float v2 = 0;
        for(int i = 0; i < input_dimension; i++){
            v2 += inputs[i] * inputs[i];
        }
        v2 = v2 / input_dimension + 1e-6f;

        float v = (float) Math.sqrt(v2);

        float[] x_ = new float[input_dimension];
        for (int i = 0; i < input_dimension; i++) {
            x_[i] = inputs[i] / v;
            outputs[i] = w[i] * x_[i] + b[i];
        }

        return new float[][]{outputs, new float[]{v, v2}, x_};
    }


    @Override
    public float[] forward(float[] inputs) {
        return forward_list(inputs)[0];
    }



    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[][] outputsList = forward_list(inputs);
        float[] ol2 = outputsList[1];
        float v = ol2[0];
        float v2 = ol2[1];
        float[] x_ = outputsList[2];

        int m = input_dimension;

        float var0 = (float) Math.pow(v2, -1.5);

        float[] DL_dX_ = new float[m];
        float DL_Dv2 = 0;
        for(int i = 0; i < m; i++) {
            DL_dX_[i] = deltas[i] * w[i];
            DL_Dv2 += DL_dX_[i] * inputs[i];
        }
        DL_Dv2 *= - var0 / m;


        float[] w_b_deltas = new float[input_dimension * 2];
        float[] last_deltas = new float[input_dimension];

        for(int i = 0; i < m; i++) {
            w_b_deltas[i] = deltas[i] * x_[i];          //dw  检验通过
            w_b_deltas[input_dimension + i] = deltas[i];

            last_deltas[i] = DL_dX_[i] / v + DL_Dv2 * inputs[i];
        }


        return new float[][]{last_deltas, w_b_deltas};
    }


    @Override
    public void upgradeWeight(float[] weightDeltas) {
        for(int i = 0; i < input_dimension * 2; i++){
            if(i < input_dimension)
                w[i] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[i], i);
            else
                b[i - input_dimension] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[i], i);
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

        w = SaveData.getsFloatArrays(in.readLine());
        b = SaveData.getsFloatArrays(in.readLine());
    }
}
