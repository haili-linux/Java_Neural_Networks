package haili.deeplearn.model;

import haili.deeplearn.function.Fuction;
import haili.deeplearn.function.MSELoss;
import haili.deeplearn.model.layer.Layer;

import java.util.ArrayList;
import java.util.Arrays;

public class Sequential {

    int input_width, input_height;
    int input_Dimension;

    public Fuction Loss_Function = new MSELoss();
    public float learn_rate = 1e-4f;

    ArrayList<Layer> layers = new ArrayList<>();



    public Sequential(int input_width, int input_height, int input_Dimension){
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_Dimension = input_Dimension;
    }


    public void addLayer(Layer layer){
        layer.learn_rate = learn_rate;

        //layer初始化
        if(layer.input_dimension == 0 || layer.input_height == 0 || layer.input_width == 0){
            if(layers.size() == 0){
                layer.init(input_width, input_height, input_Dimension);
            } else {
                Layer lastLayer = layers.get(layers.size()-1);
                layer.init(lastLayer.output_width, lastLayer.output_height, lastLayer.output_dimension);
            }
        }
        layers.add(layer);
    }

    public void setLoss_Function(Fuction loss_Function){
        this.Loss_Function = loss_Function;
    }

    public void setLearn_rate(float learn_rate){
        this.learn_rate = learn_rate;
        for (Layer layer: layers){
            layer.learn_rate = learn_rate;
        }
    }

    public float[] forward(float[] inputs){
        float[] out = layers.get(0).forward(inputs);
        for(int i = 1; i < layers.size(); i++){
            out = layers.get(i).forward(out);
        }
        return out;
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

    public void backward(float[] x_train, float[] y_train){
        ArrayList<float[]> output = forward_list(x_train);

        float[] delta = lossDelta(output.get(output.size()-1), y_train);

        for(int i = output.size()-1; i > 0; i--)
            delta = layers.get(i).backward(output.get(i-1), output.get(i), delta);

        layers.get(0).backward(x_train, output.get(0), delta);

    }


    //计算loss梯度
    private float[] lossDelta(float[] out, float[] y_train){

        float[] delta = new float[y_train.length];
        for(int i = 0; i < y_train.length; i++){
            delta[i] = Loss_Function.f_derivative(out[i], y_train[i]);
        }

        return delta;
    }


    public float loss(float[][] x, float[][] y){
        float loss = 0;
        for (int i = 0; i < x.length; i++){
            float[] oi = forward(x[i]);
            for (int j = 0; j < oi.length; j ++){
                loss += new MSELoss().f(oi[j], y[i][j]);
            }
        }
        return loss / x.length;
    }
}
