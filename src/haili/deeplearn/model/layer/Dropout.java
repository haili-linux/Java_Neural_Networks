package haili.deeplearn.model.layer;

import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Random;

public class Dropout extends Layer{

    double p;
    float ap;
    public Dropout(double p){
        this.id = 11;
        if(p <= 0 || p >= 1)
            p = 0.1;
        this.p = p;

        ap = (float) (1.0 / (1.0 - p));
    }

    @Override
    public float[] forward(float[] inputs) {
        if(train)
            return forward_train(inputs)[1];
        else
            return inputs;
    }

    public float[][] forward_train(float[] inputs) {
        if(hiddenLayerOutputMap.containsKey(inputs)){
            float[][] r = (float[][]) hiddenLayerOutputMap.get(inputs);
            hiddenLayerOutputMap.remove(inputs);
            return r;

        } else {
            float[] deltas = new float[input_dimension];
            Random random = new Random();
            for (int i = 0; i < inputs.length; i++) {
                double rp = random.nextDouble();
                if (rp < p) {
                    inputs[i] = 0;
                    deltas[i] = 0;
                } else {
                    inputs[i] *= ap;
                    deltas[i] = ap;
                }
            }
            float[][] r = new float[][]{deltas, inputs};
            hiddenLayerOutputMap.put(inputs, r);
            return r;
        }
    }


    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[][] r = forward_train(inputs);

        for(int i= 0; i < input_dimension; i++)
            r[0][i] *= deltas[i];

        return r;
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

        pw.println(SaveData.sFloat("p", (float) p));

    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

        p = SaveData.getSFloat(in.readLine());
        ap = (float) (1.0 / (1.0 - p));
    }
}
