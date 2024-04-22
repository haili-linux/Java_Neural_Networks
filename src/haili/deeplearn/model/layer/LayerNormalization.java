package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.Relu;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

public class LayerNormalization extends Layer {

    public float[] w, b;

    public LayerNormalization(int input_dimension){
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

    private Function actf(){
        return  new Function(){
            @Override
            public float f(float x) {
               if(x > 0)
                   return x;
               else
                   return -x;
            }

            @Override
            public float f_derivative(float out) {
                if(out >= 0)
                    return 1;
                else
                    return  -1;
            }
        };
    }


    // output max: input_dimension
    public float[][] forward_list(float[] inputs) {
        float[] outputs = new float[output_dimension];
        int m = input_dimension;
        float ex = 0;
        for(float xi : inputs)
            ex += xi;

        //均值
        ex = ex / m;

        float d2 = 0;
        for(float xi : inputs) {
            float var0 = xi - ex;
            d2 += var0 * var0;
        }

        //System.out.println("dx " + d2 / m);
        //方差
        d2 = d2 / m + 1e-6f;
        float d = (float) Math.sqrt(d2);

        float[] x_ = new float[m];
        for(int i = 0; i < m; i++){
            x_[i] = (inputs[i] - ex) / d;
            outputs[i] = w[i] * x_[i] + b[i];
        }


        return new float[][]{outputs, new float[]{ex, d, d2}, x_};
    }


    @Override
    public float[] forward(float[] inputs) {
        return  (float[]) forward_list(inputs)[0];
    }



    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[][] ol = forward_list(inputs);
        float[] ex_d = ol[1];
        float ex = ex_d[0];
        float d = ex_d[1];
        float d2 = ex_d[2];
        float[] x_ =  ol[2];

        float[] last_deltas = new float[input_dimension];
        float[] w_b_deltas = new float[input_dimension * 2];

        int m = input_dimension;
        //d2/dyi //方差的平方对xi的梯度
//        float[] Dd2_Dyi = new float[m];
//
//
//        double var3 =  (m * m);
//        double var1 = (m - 1) / var3;
//        for(int i = 0; i < m; i++){
//            for (int j = 0; j < m; j++) {
//                if(i == j){
//                    Dd2_Dyi[i] += (inputs[j] - ex) * var1;
//                } else {
//                    Dd2_Dyi[i] += -(inputs[j] - ex) / var3;
//                }
//            }
//            //Dd2_Dyi[i] *= 2;
//            //计算Dd2_Dyi已经通过测试
//        }
//        //System.out.println(" Dd2_Dyi 0 " + Arrays.toString(Dd2_Dyi));
//        // [-0.19999999, 0.19999999]
//
//
//        double v1 = (m - 1.0) / m;
//        double v2 = - 1.0 / m;
//        for(int i = 0; i < input_dimension; i++) {
//            //float var_0 = inputs[i] - ex;
//
//            w_b_deltas[i] = deltas[i] * (float)x_[i];          //dw  检验通过
//            w_b_deltas[input_dimension + i] = deltas[i];   //db  检验通过
//
//            //Error
//            for (int j = 0; j < input_dimension; j++) {
//                double v = 0;
//
//                if(i == j)
//                   v = v1;
//                else
//                    v = v2;
//
//                //v[0][0] =  1.255584051285119E-6
//                //v[0][1] = -1.255584051285119E-6
//                //v[1][0] = -1.255584051285119E-6
//                //v[1][1] =  1.255584051285119E-6
//                //double v00 = x_[i] * x_[j] / m;
//                //double v01 = v - v00;
//                //double v02 = v01 / d;
//                v = (v - x_[i] * x_[j] / m) / d;
//                last_deltas[j] += deltas[i] * w[i] * (float) v;
//            }
//
//            //带 1e-6
//            //dx0 =  [7.847352E-6, -7.847352E-6]
//            //dx1 = [-7.847352E-6,  7.847352E-6]
//            //tensorflow 结果
//            //dx0 = [ 7.6293945e-06, -7.6293945e-06]
//            //dx1 = [-7.987022e-06,   7.987022e-06]
//        }

        float[] DL_Dx_i = new float[m];
        float DL_Dd2 = 0;
        for(int i = 0; i < m; i++) {
            DL_Dx_i[i] = deltas[i] * w[i];
            DL_Dd2 +=  DL_Dx_i[i] * (inputs[i] - ex);
        }

        //   4.6565694E-7
        //tf 5.960446e-07
        DL_Dd2 *= -(float)(Math.pow(d2, -1.5)) * 0.5f; //eff
        //DL_Dd2 = 5.960446e-07f;
        //DL_Dd2 = 3.6500074e-07f;


        float var0 = 0, var1 = 0;
        for(int i = 0; i < m; i++) {
            var0 += DL_Dx_i[i];
            var1 += (float) (inputs[i] - ex);
        }
        var0 *= (float) (-1.0 / d);
        var1 = DL_Dd2 * (-2) * var1 /m;

        //   -4.9999843
        //tf -4.9999843
        float DL_Dex = var0 + var1; //验证正确

        for(int i = 0; i < m; i++) {
            w_b_deltas[i] = deltas[i] *x_[i];          //dw  检验通过
            w_b_deltas[input_dimension + i] = deltas[i];   //db  检验通过

            float v1 =  DL_Dx_i[i] / d;

            float v2 = DL_Dd2 * 2.0f * (inputs[i] - ex) / m;

            float v3 = DL_Dex / m;
            float v4 = v1  + v3 + v2;
            last_deltas[i] = v4 ; //DL_Dx_i[i] / d + DL_Dd2 * 2.0f * (inputs[i] - ex) / m + DL_Dex / m;
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
