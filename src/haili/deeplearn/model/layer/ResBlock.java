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
            output_layer.activity_function = Function.getFunctionById(this.activity_function.id);
        }

        this.activity_function = Function.getFunctionById(layer.activity_function.id);
        layer.activity_function = new Function();
        layers.add(layer);
    }


    /**
     * output
     * @param inputs inputs
     * @return 网络每层的的输出
     */
    private ArrayList<float[]> forward_list(float[] inputs){
        ArrayList<float[]> output = new ArrayList<>();
        output.add( layers.get(0).forward(inputs) );

        for(int i = 1; i < layers.size(); i++){
            output.add( layers.get(i).forward(output.get(i-1)) );
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
        if(ResConnectType == ResConnectType_Add){
            //残差连接方式0: 相加
            this.output_width = outlayer.output_width;
            this.output_height = outlayer.output_height;
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
            float[] resDeltas = new float[output_dimension];
            for(int i = 0; i < output_dimension; i++) {
                deltas[i] *= activity_function.f_derivative(output[i]);
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

        } else if(ResConnectType == ResConnectType_Concat) {
            //残差连接方式1: 拼接
            int outLayer_dimension = layers.get(layers.size()-1).output_dimension;
            float[] deltas_lastLayer = new float[outLayer_dimension];
            float[] out_lastLayer = new float[outLayer_dimension];
            int var0 = output_list.size() - 1;
            for(int i = 0; i < deltas_lastLayer.length; i++){
                deltas_lastLayer[i] = deltas[i] * activity_function.f_derivative(output[i]);
                out_lastLayer[i] = output_list.get(var0)[i];
            }

            // 输出层
            back = layers.get(var0).backward(output_list.get(var0 - 1), out_lastLayer, deltas_lastLayer);

            for (int i = output_list.size() - 2; i > 0; i--) {
                back = layers.get(i).backward(output_list.get(i - 1), output_list.get(i), back[0]);

                if (back[1] == null)
                    System.out.println(Arrays.toString(back[1]) + "   " + layers.get(i));

                System.arraycopy(back[1], 0, w_deltas, index, back[1].length);
                index += back[1].length;
            }

            back = layers.get(0).backward(inputs, output_list.get(0), back[0]);

            for(int i = 0; i < back[0].length; i++)
                back[0][i] += deltas[outLayer_dimension + i];

            System.arraycopy(back[1], 0, w_deltas, index, back[1].length);
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
                System.out.println(" ResBlock " + this.toString() + ": 输入和输出的维度不一致。this.input_dimension != this.output_dimension!");
                System.exit(0);
            } else {
                //跨层连接
                out = MatrixUtil.add(out, inputs);

                for(int i = 0; i < out.length; i++)
                    out[i] = activity_function.f(out[i]);
            }
        } else if(ResConnectType == ResConnectType_Concat){
            for(int i = 0; i < out.length; i++)
                out[i] = activity_function.f(out[i]);
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

    @Override
    public int getWeightNumber() {
        int number = 0;
        for(Layer layer: layers)
            number += layer.getWeightNumber();

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
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("input_Dimension", input_dimension));
        pw.println(SaveData.sInt("input_width", input_width));
        pw.println(SaveData.sInt("input_height", input_height));

        pw.println(SaveData.sInt("output_Dimension", output_dimension));
        pw.println(SaveData.sInt("output_width", output_width));
        pw.println(SaveData.sInt("output_height", output_height));

        pw.println(SaveData.sFloat("learn_rate", learn_rate));

        for (Layer layer : layers)
            layer.saveInFile(pw);
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {

        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

        learn_rate = SaveData.getSFloat(in.readLine());

        String line = null;
        while ((line = in.readLine()) != null){
            Layer layer = getLayerById(SaveData.getSInt(line));
            layer.initByFile(in);
            layers.add(layer);
        }

        setLearn_rate(learn_rate);
    }

}
