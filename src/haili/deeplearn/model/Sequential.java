package haili.deeplearn.model;

import haili.deeplearn.function.Fuction;
import haili.deeplearn.function.MSELoss;
import haili.deeplearn.model.layer.Conv2D;
import haili.deeplearn.model.layer.Dense;
import haili.deeplearn.model.layer.Layer;
import haili.deeplearn.model.layer.Pooling2D;
import haili.deeplearn.utils.SaveData;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Sequential extends SaveData {

    public String EXPLAIN = "";


    int input_width, input_height;
    int input_Dimension;

    public Fuction Loss_Function = new MSELoss();

    public float loss = 0;

    public float learn_rate = 1e-4f;

    public ArrayList<Layer> layers = new ArrayList<>();



    public Sequential(int input_width, int input_height, int input_Dimension){
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_Dimension = input_Dimension;
    }

    public Sequential(String fileName){
        try {
            InitByFile(fileName);
        } catch (Exception exception) {
            exception.printStackTrace();
        }
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
        return this.loss = loss / x.length;
    }


    public void saveInFile(String fileName) throws Exception{
        File file = new File(fileName);
        if (file.exists()) {
            String p1 = fileName.substring(0, fileName.lastIndexOf("."));
            for (int i = 0; ; i++) {
                file = new File(fileName = p1 + "_" + i + ".log");
                if (!file.exists()) break;
            }
        }
        try { boolean newFile = file.createNewFile(); } catch (Exception ignored) { }

        if(file.isFile()) {
            FileWriter fw = null;
            try { fw = new FileWriter(file, true); } catch (IOException e) { e.printStackTrace(); return; }

            PrintWriter pw = new PrintWriter(fw);

            pw.println("explain: Sequential:" + EXPLAIN);

            pw.println(sInt("input_Dimension", input_Dimension));
            pw.println(sInt("input_width", input_width));
            pw.println(sInt("input_height", input_height));

            pw.println(sFloat("loss", loss));
            pw.println(sFloat("learn_rate", learn_rate));
            pw.println(sInt("LossFunction", Loss_Function.id));

            for (Layer layer : layers)
                layer.saveInFile(pw);

            pw.flush();
            try {
                fw.flush();
                pw.close();
                fw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }



    private void InitByFile(String fileName) throws Exception{
        File file = new File(fileName);
        if(file.isFile())
            try {
                FileReader fileReader = null;
                fileReader = new FileReader(file);
                BufferedReader in = new BufferedReader(fileReader);

                String line = in.readLine();
                EXPLAIN = line.substring(20);

                input_Dimension = getSInt(in.readLine());
                input_width = getSInt(in.readLine());
                input_height = getSInt(in.readLine());

                loss = getSFloat(in.readLine());
                learn_rate = getSFloat(in.readLine());
                Loss_Function = Fuction.getFunctionById(getSInt(in.readLine()));

                while ((line=in.readLine()) != null){
                    Layer layer = getLayerById(getSInt(line));
                    layer.InitByFile(in);
                    layers.add(layer);
                }

                in.close();
                fileReader.close();
            }catch(Exception e){
                e.printStackTrace();
            }
    }

    private Layer getLayerById(int id){
        Layer layer;
        switch (id){
            case 1: layer = new Dense(1, new Fuction()); break;
            case 2: layer = new Conv2D(1,1,1,new Fuction()); break;
            case 3: layer = new Pooling2D(1,1); break;
            default: layer = new Layer(); break;
        }
        return layer;
    }

    @Override
    public String toString() {
        return "Sequential{" +
                "input_width=" + input_width +
                ", input_height=" + input_height +
                ", input_Dimension=" + input_Dimension +
                ", Loss_Function=" + Loss_Function +
                ", learn_rate=" + learn_rate +
                ", layers=" + layers +
                '}';
    }
}
