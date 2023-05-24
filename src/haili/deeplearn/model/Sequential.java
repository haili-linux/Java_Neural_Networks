package haili.deeplearn.model;

import haili.deeplearn.function.Fuction;
import haili.deeplearn.function.MSELoss;
import haili.deeplearn.model.layer.Layer;

import java.util.ArrayList;

public class Sequential {

    public Fuction Loss_Function;
    public float learn_rate = 1e-4f;

    ArrayList<Layer> layers = new ArrayList<>();

    public void addLayer(Layer layer){
        layer.learn_rate = learn_rate;
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

    public float[] out(float[] inputs){
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
    private ArrayList<float[]> out_list(float[] inputs){
        ArrayList<float[]> output = new ArrayList<>();
        output.add( layers.get(0).forward(inputs) );

        for(int i = 1; i < layers.size(); i++){
            output.add( layers.get(i).forward(output.get(i-1)) );
        }
        return output;
    }

    public void back(float[] x, float[] y){
        ArrayList<float[]> output = out_list(x);

        float[] delta = new float[y.length];
        float[] lastOut = output.get(output.size()-1);
        for(int i = 0; i < y.length; i++){
            delta[i] = Loss_Function.f_derivative(lastOut[i], y[i]);
        }

        for(int i = output.size()-1; i > 0; i--)
            delta = layers.get(i).backward(output.get(i-1), output.get(i), delta);

        layers.get(0).backward(x, output.get(0), delta);

    }

    public float loss(float[][] x, float[][] y){
        float loss = 0;
        for (int i = 0; i < x.length; i++){
            float[] oi = out(x[i]);
            for (int j = 0; j < oi.length; j ++){
                loss += new MSELoss().f(oi[j], y[i][j]);
            }
        }

        return loss / x.length;
    }
}
