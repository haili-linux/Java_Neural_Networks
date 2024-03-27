package haili.deeplearn.model.layer;



import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.Neuron;
import haili.deeplearn.function.Function;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;


public class Dense extends Layer{

    public Neuron[] neurons;

    //public Function activity_function;

    public Dense(int input_Dimension, int output_Dimension, Function activation){
        id = 1;
        this.activity_function = activation;
        this.output_dimension = output_Dimension;
        init(-1, -1, input_Dimension);
    }


    public Dense(int output_Dimension, Function activation){
        id = 1;
        this.output_dimension = output_Dimension;
        this.activity_function = activation;
        //neurons = new Neuron[output_Dimension];
    }

    public Dense(int input_width, int input_height, int output_width, int output_height, int output_dimension , Function activation){
        this.input_dimension = input_width * input_height;
        this.output_width = output_width;
        this.output_height = output_height;
        id = 1;
        this.activity_function = activation;
        this.output_dimension = output_dimension;
        init(input_width, input_height, this.input_dimension);
    }

//    public Dense(int output_width, int output_height, Function activation){
//        this.input_dimension = input_width * input_height;
//        this.output_width = output_width;
//        this.output_height = output_height;
//        id = 1;
//        this.activity_function = activation;
//        this.output_dimension = output_width * output_height;
//        init(-1, -1, this.input_dimension);
//    }

    @Override
    public void init(int input_width, int input_height, int input_dimension){


        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_dimension;

        if(neurons != null)
            return;

        neurons = new Neuron[output_dimension];

        for (int i = 0; i < neurons.length; i++)
            neurons[i] = new Neuron(input_dimension, activity_function);
    }


    @Override
    public float[] forward(float[] inputs) {
        float[] output = new float[output_dimension];

        for (int i = 0; i < output.length; i++)
            output[i] = neurons[i].out_notSave(inputs);

        return output;
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
        for(int i = 0; i < neurons.length; i++){

            deltas[i] *= neurons[i].ACT_function.f_derivative(output[i]);
            w_b_deltas[index] = deltas[i];
            index++;

            for(int j = 0; j < neurons[i].w.length; j++) {
                last_layer_deltas[j] += deltas[i] * neurons[i].w[j];

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

        for(int i = 0; i < neurons.length; i++){
            neurons[i].b -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
            index++;

            for(int j = 0; j < neurons[i].w.length; j++) {
                neurons[i].setW(j, neurons[i].w[j] - learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index));
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
            WeightNumber =(input_dimension + 1) * output_dimension;

        return WeightNumber;
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception{
        pw.println(SaveData.sInt("Layer_ID", id));
        pw.println(SaveData.sInt("input_dimension", input_dimension));
        pw.println(SaveData.sInt("output_dimension", output_dimension));

        for(int i = 0; i < neurons.length; i++){
            pw.println(SaveData.sInt("neurons[" + i + "].Act_Function_ID", neurons[i].ACT_function.id));
            pw.println(SaveData.sFloat("neurons[" + i + "].bias", neurons[i].b));
            for (int j = 0; j < neurons[i].w.length; j++)
                pw.println(SaveData.sFloat("neurons[" + i + "].w[" + j + "]", neurons[i].w[j]));
        }
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception{
        input_width = input_height = -1;

        input_dimension = SaveData.getSInt(in.readLine());
        output_dimension = SaveData.getSInt(in.readLine());

        neurons = new Neuron[output_dimension];
        for(int i = 0; i < output_dimension; i++){
            Neuron ni = new Neuron();
            ni.input_dimension = input_dimension;

            int actFunctionID = SaveData.getSInt(in.readLine());
            ni.ACT_function = Function.getFunctionById(actFunctionID);

            ni.b = SaveData.getSFloat(in.readLine());
            ni.w = new float[input_dimension];
            for (int j = 0; j < input_dimension; j++)
                ni.w[j] = SaveData.getSFloat(in.readLine());

            neurons[i] = ni;
        }
    }

    @Override
    public String toString() {
        return "Dense{" +
                "activation=" + activity_function +
                ", input_dimension=" + input_dimension +
                ", input_width=" + input_width +
                ", input_height=" + input_height +
                ", output_dimension=" + output_dimension +
                ", output_width=" + output_width +
                ", output_height=" + output_height +
                '}';
    }
}
