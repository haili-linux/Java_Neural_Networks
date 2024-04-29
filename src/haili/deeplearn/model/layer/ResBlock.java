package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

public class ResBlock extends Layer{

    // 残差连接方式0: 相加，ResBlock的输入的shape和最后一层输出的shape相同。经过最后一层激活函数
    public static final int ResConnectType_Add = 0;

    // 残差连接方式1: 拼接，会失去长宽信息。不经过最后一层激活函数
    public static final int ResConnectType_Concat = 1;

    public int ResConnectType;

    public ArrayList<Layer> layers = new ArrayList<>();

    public ResBlock(int ResConnectType){
        this.id = 5;
        this.ResConnectType = ResConnectType;
    }

    public ResBlock(int input_dimension, int ResConnectType){
        this.id = 5;
        this.input_dimension = input_dimension;
        this.ResConnectType = ResConnectType;
    }

    public void addLayer(Layer layer){
        if(layers.isEmpty()){
            this.output_dimension = layer.output_dimension;
        } else {
            Layer output_layer = layers.get(layers.size() - 1);
            output_layer.activation_function = Function.getFunctionById(this.activation_function.id);
        }

        this.activation_function = Function.getFunctionById(layer.activation_function.id);
        layer.activation_function = new Function();
        layers.add(layer);
    }


    public void setSaveHiddenLayerOutput(boolean b){
        this.saveHiddenLayerOutput = b;
        for (Layer layer: layers)
            layer.setSaveHiddenLayerOutput(b);
    }

    public void clearHiddenLayerOutput(){
        this.hiddenLayerOutputMap.clear();
        for (Layer layer: layers)
            layer.clearHiddenLayerOutput();
    }


    /**
     * output
     * @param inputs inputs
     * @return 网络每层的的输出
     */
    private ArrayList<float[]> forward_list(float[] inputs){
        ArrayList<float[]> output;
        // 如果有缓存，从缓存读取，不需要重新计算
        if(saveHiddenLayerOutput && hiddenLayerOutputMap.containsKey(inputs)) {
            output = (ArrayList<float[]>) hiddenLayerOutputMap.get(inputs);
            if(output != null)
                return output;
        }

        output = new ArrayList<>();
        output.add( layers.get(0).forward(inputs) );

        for(int i = 1; i < layers.size(); i++){
            output.add( layers.get(i).forward(output.get(i-1)) );
        }

        // 保存中间输出
        if(saveHiddenLayerOutput){
            hiddenLayerOutputMap.put(inputs, output);
        }
        return output;
    }

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        //super.init(input_width, input_height, input_Dimension);
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_Dimension;

        Layer layer_0 = layers.get(0);
        layer_0.init(input_width, input_height, input_Dimension);

        for(int i = 1; i < layers.size(); i++){
            Layer lastLayer = layers.get(i - 1);
            Layer layer_i = layers.get(i);
            layer_i.init(lastLayer.output_width, lastLayer.output_height, lastLayer.output_dimension);
        }

        Layer outlayer = layers.get(layers.size() - 1);
        this.output_width = outlayer.output_width;
        this.output_height = outlayer.output_height;
        if(ResConnectType == ResConnectType_Add){
            //残差连接方式0: 相加
            this.output_dimension = outlayer.output_dimension;
        } else if(ResConnectType == ResConnectType_Concat){
            //残差连接方式1: 拼接
            this.output_dimension = outlayer.output_dimension + this.input_dimension;
        }
    }



    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        ArrayList<float[]> output_list = forward_list(inputs);

        float[] w_deltas = new float[getWeightNumber()];
        int index = 0;


        float[][] back = new float[2][];
        //back[0] = deltas;

        if(ResConnectType == ResConnectType_Add){
            //残差连接方式0: 相加
            float[] resDeltas = new float[output.length];
            for(int i = 0; i < output.length; i++) {
                deltas[i] *= activation_function.f_derivative(output[i]);
                resDeltas[i] = deltas[i];
            }

            back[0] = deltas;

            for(int i = output_list.size()-1; i > 0; i--) {
                back = layers.get(i).backward(output_list.get(i - 1), output_list.get(i), back[0]);

                if(back[1] == null) {
                    System.out.println(Arrays.toString(back[1]) + "   " + layers.get(i));
                    System.exit(0);
                }

                System.arraycopy(back[1], 0, w_deltas, index, back[1].length);
                index += back[1].length;
            }

            back = layers.get(0).backward(inputs, output_list.get(0), back[0]);

            //加上残连接的梯度
            back[0] = MatrixUtil.add(back[0], resDeltas);

            System.arraycopy(back[1], 0, w_deltas, index, back[1].length);

        } else
            if (ResConnectType == ResConnectType_Concat) {
                //残差连接方式1: 拼接
                float[] input_lastLayer;
                if (layers.size() == 1)
                    input_lastLayer = inputs;
                else
                    input_lastLayer = output_list.get(output_list.size() - 2);

                float[] out_lastLayer = output_list.get(output_list.size() - 1);
                int outLayer_dimension = out_lastLayer.length;
                float[] deltas_lastLayer = new float[outLayer_dimension];

                int var0 = output_list.size() - 1;
                float[] out_last = output_list.get(var0);
                for (int i = 0; i < deltas_lastLayer.length; i++) {
                    deltas_lastLayer[i] = deltas[i] * activation_function.f_derivative(output[i]);
                    out_lastLayer[i] = out_last[i];
                }

                // 输出层
                back = layers.get(var0).backward(input_lastLayer, out_lastLayer, deltas_lastLayer);
                System.arraycopy(back[1], 0, w_deltas, index, back[1].length);
                index += back[1].length;

                //layers.size() == 1 时，输入层等于输出层
                if (layers.size() > 1) {
                    for (int i = output_list.size() - 2; i > 0; i--) {
                        back = layers.get(i).backward(output_list.get(i - 1), output_list.get(i), back[0]);
                        System.arraycopy(back[1], 0, w_deltas, index, back[1].length); //储存w的梯度
                        index += back[1].length;
                    }

                    //输入层
                    back = layers.get(0).backward(inputs, output_list.get(0), back[0]);
                    System.arraycopy(back[1], 0, w_deltas, index, back[1].length);
                }

                for (int i = 0; i < back[0].length; i++)
                    back[0][i] += deltas[outLayer_dimension + i];
            }

        return new float[][]{back[0], w_deltas};
    }

    @Override
    public float[] forward(float[] inputs) {
        float[] out = layers.get(0).forward(inputs);

        for(int i = 1; i < layers.size(); i++)
            out = layers.get(i).forward(out);

        if(ResConnectType == ResConnectType_Add){
            //残差连接方式0: 相加
            if(this.input_dimension != this.output_dimension){
                System.out.println(" ResBlock " + this.toString() + ": \n输入和输出的维度不一致。this.input_dimension != this.output_dimension!  input:"  );
                System.exit(0);
            } else {
                //跨层连接
                out = MatrixUtil.add(out, inputs);

                for(int i = 0; i < out.length; i++)
                    out[i] = activation_function.f(out[i]);
            }
        } else if(ResConnectType == ResConnectType_Concat){
            for(int i = 0; i < out.length; i++)
                out[i] = activation_function.f(out[i]);
            //残差连接方式1: 拼接 out = { out0, out1, out2, ..., outN, input0, intput1, ..., inputN}
            out = MatrixUtil.combine(out, inputs);
        }

        return out;
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) {
        int index = 0;
        for(int i = layers.size()-1; i >= 0; i--) {
            int w_number = layers.get(i).getWeightNumber();
            if (w_number > 0) {
                float[] w_delta = new float[w_number];
                System.arraycopy(weightDeltas, index, w_delta, 0, w_number);
                index += w_number;
                layers.get(i).upgradeWeight(w_delta);
            }
        }
    }

    int WeightNumber = -1;
    @Override
    public int getWeightNumber() {
        if(WeightNumber == -1) {
            int number = 0;
            for (Layer layer : layers)
                number += layer.getWeightNumber();

            WeightNumber = number;
        }
        return WeightNumber;
    }

    @Override
    public int getWeightNumber_Train() {
        int number = 0;
        for (Layer layer : layers)
            number += layer.getWeightNumber_Train();

        return number;
    }

    @Override
    public void setLearn_rate(float learn_rate){
        this.learn_rate = learn_rate;
        for (Layer layer: layers){
            layer.setLearn_rate(learn_rate);
        }
    }

    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer){
        for (Layer layer: layers){
            layer.setDeltaOptimizer(deltaOptimizer);
        }
    }

    @Override
    public void setTrain(boolean train) {
        this.train = train;
        for (Layer layer: layers){
            layer.setTrain(this.train);
        }
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("ResConnectType", ResConnectType));
        pw.println(SaveData.sInt("input_Dimension", input_dimension));
        pw.println(SaveData.sInt("input_width", input_width));
        pw.println(SaveData.sInt("input_height", input_height));

        pw.println(SaveData.sInt("output_Dimension", output_dimension));
        pw.println(SaveData.sInt("output_width", output_width));
        pw.println(SaveData.sInt("output_height", output_height));

        pw.println(SaveData.sFloat("learn_rate", learn_rate));

        pw.println(SaveData.sInt("layer_number", layers.size()));
        for (Layer layer : layers)
            layer.saveInFile(pw);
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        ResConnectType = SaveData.getSInt(in.readLine());
        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

        learn_rate = SaveData.getSFloat(in.readLine());

        int layer_num = SaveData.getSInt(in.readLine());

        String line = null;
        while ((line = in.readLine()) != null){
            Layer layer = getLayerById(SaveData.getSInt(line));
            layer.initByFile(in);
            layers.add(layer);

            if(layers.size() == layer_num)
                break;
        }

        setLearn_rate(learn_rate);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder= new StringBuilder();
        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        if(this.ResConnectType == ResConnectType_Add)
            name += "-Add";
        else
            name += "-Concat";

        char[] c0 = new char[32 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "(" + output_width + ", " + output_height + ", " + output_dimension + ")";

        int v0 = 25 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');
        int param = getWeightNumber_Train();

        stringBuilder.append(name).append(c0).append(output_shape).append(c1).append(param);

        for (Layer layer: layers){
            stringBuilder.append("\n ").append(layer.toString());
        }

        return stringBuilder.toString();
    }
}
