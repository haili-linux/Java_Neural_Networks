package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;


public class Dense extends Layer{

    public float[][] w;
    public float[] bias;

    public Dense(int input_Dimension, int output_Dimension, Function activation){
        id = 1;
        this.activation_function = activation;
        this.output_dimension = output_Dimension;
        init(input_Dimension, 1, input_Dimension);
    }

    public Dense(int output_Dimension){
        id = 1;
        this.output_dimension = output_Dimension;
    }


    public Dense(int output_Dimension, Function activation){
        id = 1;
        this.output_dimension = output_Dimension;
        this.activation_function = activation;
    }

    public Dense(int output_Dimension, Function activation, boolean use_bias){
        id = 1;
        this.output_dimension = output_Dimension;
        this.activation_function = activation;
        this.use_bias = use_bias;
    }

    public Dense(int input_width, int input_height, int output_width, int output_height, int output_dimension , Function activation){
        this.input_dimension = input_width * input_height;
        this.output_width = output_width;
        this.output_height = output_height;
        id = 1;
        this.activation_function = activation;
        this.output_dimension = output_dimension;
        init(input_width, input_height, this.input_dimension);
    }

    public Dense(int input_width, int input_height, int output_width, int output_height, int output_dimension , Function activation, boolean use_bias){
        this.input_dimension = input_width * input_height;
        this.output_width = output_width;
        this.output_height = output_height;
        id = 1;
        this.activation_function = activation;
        this.output_dimension = output_dimension;
        this.use_bias = use_bias;
        init(input_width, input_height, this.input_dimension);
    }

    @Override
    public void init(int input_width, int input_height, int input_dimension){
        if(w != null)
            return;

        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_dimension;

        w = new float[output_dimension][];
        bias = new float[output_dimension];

        for(int i = 0; i < output_dimension; i++)
            w[i] = GaussRandomArrays(input_dimension);
    }


    @Override
    public float[] forward(float[] inputs) {
        float[] outputs = new float[output_dimension];

        for (int i = 0; i < output_dimension; i++) {

            outputs[i] = bias[i];

            for (int j = 0; j < input_dimension; j++)
                outputs[i] += w[i][j] * inputs[j];

            outputs[i] = activation_function.f( outputs[i]);
        }

        return outputs;
    }


    /**
     * 反向传播
     * @param inputs 本层输入
     * @param output 本层输出
     * @param deltas 下一层传回的梯度，对应本层每个神经元
     * @return [0]传给上一层的梯度，对应上一层每个神经元, [1]:本层参数的梯度
     */
    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {

        float[] w_b_deltas = new float[getWeightNumber()];
        float[] last_layer_deltas = new float[input_dimension];

        int index = 0;
        for(int i = 0; i < output_dimension; i++){
            deltas[i] *= activation_function.f_derivative(output[i]);

            if(use_bias)
                w_b_deltas[index] = deltas[i];  //delta_bias

            index++;

            for(int j = 0; j < input_dimension; j++) {
                last_layer_deltas[j] += deltas[i] * w[i][j];
                w_b_deltas[index] = deltas[i] * inputs[j];

                index++;
            }
        }

        return new float[][]{last_layer_deltas, w_b_deltas};
    }


    /**
     * 更新权重
     * @param weightDeltas backward()返回的第2个
     */
    @Override
    public void upgradeWeight(float[] weightDeltas) {
        int index = 0;
        for (int i = 0; i < output_dimension; i++) {
            bias[i] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
            index++;

            for (int j = 0; j < input_dimension; j++) {
                w[i][j] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
                index++;
            }
        }
    }

    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        deltaOptimizer = deltaOptimizer.getNewObject();
        deltaOptimizer.init(getWeightNumber() );
        super.setDeltaOptimizer(deltaOptimizer);
    }


    int WeightNumber = -1;
    @Override
    public int getWeightNumber() {
        if(WeightNumber == -1)
            WeightNumber = (input_dimension + 1) * output_dimension;

        return WeightNumber;
    }

    @Override
    public int getWeightNumber_Train() {
        if(use_bias)
            return getWeightNumber();
        else
            return getWeightNumber() - output_dimension;
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception{
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("input_width", input_width));
        pw.println(SaveData.sInt("input_height", input_height));
        pw.println(SaveData.sInt("input_dimension", input_dimension));

        pw.println(SaveData.sInt("output_width", output_width));
        pw.println(SaveData.sInt("output_height", output_height));
        pw.println(SaveData.sInt("output_dimension", output_dimension));

        pw.println(SaveData.sInt("activation", activation_function.id));

        int use_bias_int = 0;
        if(use_bias) use_bias_int = 1;
        pw.println(SaveData.sInt("use_bias", use_bias_int));


        pw.println(SaveData.sFloatArrays("bias", bias));


        for(int i = 0; i < w.length; i++){
            pw.println(SaveData.sFloatArrays("w[" + i + "]", w[i]));
        }
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception{

        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());
        input_dimension = SaveData.getSInt(in.readLine());

        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());
        output_dimension = SaveData.getSInt(in.readLine());

        activation_function = Function.getFunctionById(SaveData.getSInt(in.readLine()));

        int use_bias_int = SaveData.getSInt(in.readLine());
        if(use_bias_int == 0) this.use_bias = false;

        bias = SaveData.getsFloatArrays(in.readLine());

        w = new float[output_dimension][];
        for(int i = 0 ; i < output_dimension; i++)
            w[i] = SaveData.getsFloatArrays(in.readLine());

    }

}
