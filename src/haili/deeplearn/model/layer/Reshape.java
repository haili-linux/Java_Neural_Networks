package haili.deeplearn.model.layer;

import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class Reshape extends Layer{

    public Reshape(int output_width, int output_height, int output_dimension){
        this.id = 10;
        this.output_width = output_width;
        this.output_height = output_height;
        this.output_dimension = output_dimension;
    }

    public Reshape(int output_width, int output_height){
        this.id = 10;
        this.output_width = output_width;
        this.output_height = output_height;
    }

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        this.output_dimension = input_Dimension;
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

    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());
    }

}
