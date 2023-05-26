package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizer;
import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class Layer implements LayerInterface{

    public int id = -1;

    public float learn_rate = 1e-4f;

    public int input_dimension = 0;
    public int input_width = 0, input_height = 0;

    public int output_dimension = 0;
    public int output_width = 0, output_height = 0;

    protected BaseOptimizerInterface deltaOptimizer = new BaseOptimizer();

    @Override
    public void init(int input_width, int input_height, int input_Dimension) { }

    @Override
    public float[] forward(float[] inputs) {
        return new float[0];
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        return new float[0][];
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) { }

    //返回参数个数
    public int getWeightNumber(){ return 0; }

    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        this.deltaOptimizer = deltaOptimizer;
    }

    public void initByFile(BufferedReader in) throws Exception{

    }

    public void saveInFile(PrintWriter pw) throws Exception{ }

}
