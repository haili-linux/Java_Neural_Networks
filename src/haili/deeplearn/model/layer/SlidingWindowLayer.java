package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class SlidingWindowLayer extends Layer{

    Layer model;

    public SlidingWindowLayer(int one_input_vector_dimension, Layer model){
        this.id = 12;
        this.model = model;

        model.init(one_input_vector_dimension, 1, one_input_vector_dimension);

        this.input_width = one_input_vector_dimension;
        this.output_width = model.output_dimension;
        this.output_dimension = model.output_dimension;
    }

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {  }

    public Object[] forwardList(float[] inputs){
        // 当前输入的seq数量
        int seqLen = inputs.length / input_width;
        float[] outputs = new float[seqLen * output_width];
        float[][] inputs_ = new float[seqLen][];
        float[][] outputs_ = new float[seqLen][];

        for(int i = 0; i < seqLen; i++){
            inputs_[i] =  new float[input_width];
            System.arraycopy(inputs, i * input_width, inputs_[i], 0, input_width);

            outputs_[i] = model.forward(inputs_[i]);

            System.arraycopy(outputs_[i], 0, outputs, i * output_width, outputs_[i].length);
            //for(int j = 0; j < outputs_[i].length; j++)
            //    outputs[i * output_width + j] = outputs_[i][j];
        }

        return  new Object[]{outputs, inputs_, outputs_};
    }

    @Override
    public float[] forward(float[] inputs) {
        return  (float[]) forwardList(inputs)[0];
    }


    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        Object[] outputsObj = forwardList(inputs);
        float[][] inputs_ = (float[][]) outputsObj[1];
        float[][] outputs_ = (float[][]) outputsObj[2];

        //float[][] inputs_deltas_ = new float[inputs_.length][];
        float[] w_deltas = new float[model.getWeightNumber()];
        float[] inputs_deltas = new float[inputs.length];

        int len = deltas.length / output_dimension;

        for(int i = 0; i < len; i++){
            float[] deltas_i = new float[output_dimension];
            System.arraycopy(deltas, i * output_dimension, deltas_i, 0, output_dimension);
            //for(int j = 0;  j < output_dimension; j++)
            //   deltas_i[j] = deltas[i * output_dimension + j];

            float[][] backs = model.backward(inputs_[i], outputs_[i], deltas_i);

            System.arraycopy(backs[0], 0, inputs_deltas, i * input_width, input_dimension);
            //for(int j = 0;  j < input_dimension; j++)
            //    inputs_deltas[i * input_width + j] = backs[0][j];

            for(int j = 0;  j < w_deltas.length; j++)
                w_deltas[j] += backs[1][j];
        }

        return new float[][]{inputs_deltas, w_deltas};
    }

    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        model.setDeltaOptimizer(deltaOptimizer);
    }

    @Override
    public int getWeightNumber() {
        return model.getWeightNumber();
    }

    @Override
    public int getWeightNumber_Train() {
        return model.getWeightNumber_Train();
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) {
        model.upgradeWeight(weightDeltas);
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("input_width", input_width));
        pw.println(SaveData.sInt("output_width", output_width));

        model.saveInFile(pw);
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        this.input_width = SaveData.getSInt(in.readLine());
        this.output_width = SaveData.getSInt(in.readLine());;
        this.output_dimension = output_width;

        model = Layer.getLayerById(SaveData.getSInt(in.readLine()));
        model.initByFile(in);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder= new StringBuilder();
        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        char[] c0 = new char[27 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "in:(N, " + input_width + ")  out:(N, " + output_dimension + ")  ";

        int v0 = 30 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');

        int param = getWeightNumber_Train();

        stringBuilder.append(name).append(c0).append(output_shape).append(c1).append(param);
        return stringBuilder.toString();
    }
}
