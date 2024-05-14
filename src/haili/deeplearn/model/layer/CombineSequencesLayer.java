package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class CombineSequencesLayer extends Layer{

    public Layer layer;

    public CombineSequencesLayer(int one_input_vector_dimension){
        this.id = 14;

        this.input_width = one_input_vector_dimension;
        this.input_height = 1;
        this.input_dimension = one_input_vector_dimension;

        this.output_width = one_input_vector_dimension;
        this.output_height = 1;
        this.output_dimension = one_input_vector_dimension;

        layer = new Dense(input_width, 1, output_width, 1, output_width, new Function(), false);
        layer.addDeepOfSequential();
    }

    public CombineSequencesLayer(Layer layer){
        this.id = 14;

        this.input_width = layer.input_width;
        this.input_height = layer.input_height;
        this.input_dimension = layer.input_dimension;

        this.output_width = layer.output_width;
        this.output_height = layer.output_height;
        this.output_dimension = layer.output_dimension;

        this.layer = layer;
        this.layer.addDeepOfSequential();
    }


    @Override
    public void init(int input_width, int input_height, int input_Dimension) {

    }


    public Object[] forwardList(float[] inputs) {
        // 当前输入的seq数量
        int seqLen = inputs.length / input_width;

        float[] outputs = new float[output_width];
        float[][] inputs_ = new float[seqLen][];
        float[][] outputs_ = new float[seqLen][];

        for (int i = 0; i < seqLen; i++) {
            inputs_[i] = new float[input_width];
            System.arraycopy(inputs, i * input_width, inputs_[i], 0, input_width);

            outputs_[i] = layer.forward(inputs_[i]);
            for(int j = 0; j < output_width; j++)
                outputs[j] += outputs_[i][j];
        }

        return new Object[]{outputs, inputs_, outputs_};
    }


    @Override
    public float[] forward(float[] inputs) {
        return (float[]) forwardList(inputs)[0];
    }


    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        Object[] outputsObj = forwardList(inputs);
        float[][] inputs_ = (float[][]) outputsObj[1];
        float[][] outputs_ = (float[][]) outputsObj[2];

        float[] w_deltas = new float[getWeightNumber()];
        float[] input_deltas = new float[inputs.length];

        for(int seqi = 0; seqi < inputs_.length; seqi++){
            float[][] backs_i = layer.backward(inputs_[seqi], outputs_[seqi], deltas);
            float[] inputs_di = backs_i[0];
            float[] wd_i = backs_i[1];

            System.arraycopy(inputs_di, 0, input_deltas, seqi * input_width, input_width);
            //for(int i = 0; i < input_width; i++)
            //    input_deltas[seqi * input_width + i] = inputs_di[i];

            for(int i = 0; i < w_deltas.length; i++)
                w_deltas[i] += wd_i[i];
        }

        return new float[][]{input_deltas, w_deltas};
    }


    @Override
    public int getWeightNumber() {
        return layer.getWeightNumber();
    }


    @Override
    public int getWeightNumber_Train() {
        return layer.getWeightNumber_Train();
    }


    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        layer.setDeltaOptimizer(deltaOptimizer);
    }


    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("input_width", input_width));

        layer.saveInFile(pw);
    }


    @Override
    public void initByFile(BufferedReader in) throws Exception {
        int one_input_vector_dimension = SaveData.getSInt(in.readLine());

        layer = Layer.getLayerById(SaveData.getSInt(in.readLine()));
        layer.initByFile(in);
        layer.addDeepOfSequential();

        this.input_width = one_input_vector_dimension;//layer.input_width;
        this.input_height = layer.input_height;
        this.input_dimension = layer.input_dimension;

        this.output_width = layer.output_width;
        this.output_height = layer.output_height;
        this.output_dimension = layer.output_dimension;


    }


    @Override
    public String toString() {
        StringBuilder stringBuilder= new StringBuilder();
        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        char[] c0 = new char[27 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "in:(N, " + input_dimension + ")  out:(1, " + output_dimension + ")  ";

        int v0 = 30 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');

        int param = getWeightNumber_Train();

        char[] c2 = new char[deepOfSequential * 2];
        Arrays.fill(c2, ' ');

        stringBuilder.append(c2).append(name).append(c0).append(output_shape).append(c1).append(param);
        return stringBuilder.toString();
    }
}
