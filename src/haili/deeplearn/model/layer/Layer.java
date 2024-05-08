package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizer;
import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.model.Sequential;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Layer implements LayerInterface{

    public int id = -1;

    public float learn_rate = 1e-4f;

    public int input_width = 1;
    public int input_height = 1;
    public int input_dimension = 1;

    public int output_width = 1;
    public int output_height = 1;
    public int output_dimension = 1;

    public Function activation_function = new Function();
    protected BaseOptimizerInterface deltaOptimizer = new BaseOptimizer();

    //使用偏置值bias
    public boolean use_bias = true;

    //部分层的forward需要区分train和eval
    public boolean train = false;

    //是否保存中间变量
    public boolean saveHiddenLayerOutput = false;

    //用于保存中间变量
    public Map<float[], Object> hiddenLayerOutputMap = new HashMap<>();

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_Dimension;

        this.output_width = input_width;
        this.output_height = input_height;
        this.output_dimension = input_Dimension;
    }

    @Override
    public float[] forward(float[] inputs) {
        return inputs;
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
        return new float[][]{deltas, new float[0]};
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) { }

    //返回参数个数
    public int getWeightNumber(){ return 0; }

    public int getWeightNumber_Train(){ return 0; }

    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        this.deltaOptimizer = deltaOptimizer;
    }

    public void setActivation_Function(Function activation){
        this.activation_function = activation;
    }

    public void initByFile(BufferedReader in) throws Exception{ }

    public void saveInFile(PrintWriter pw) throws Exception{ }

    public void setLearn_rate(float learn_rate){
        this.learn_rate = learn_rate;
    }

    public void setSaveHiddenLayerOutput(boolean b){
        this.saveHiddenLayerOutput = b;
    }

    public void setTrain(boolean train){
        this.train = train;
    }

    public void clearHiddenLayerOutput(){
        this.hiddenLayerOutputMap.clear();
    }

    //生成gauss分布数组，均值0，
    public static float[] GaussRandomArrays(int length){
        Random random = new Random();
        float var0 = (float) Math.sqrt(length);
        float[] array = new float[length];
        for (int i = 0; i < length; i++)
            array[i] = (float) random.nextGaussian() / var0;

        return array;
    }

    public static Layer getLayerById(int id){
        Layer layer;
        switch (id){
            case 0: layer = new Sequential(1, 1, 1); break;
            case 1: layer = new Dense(1, new Function()); break;
            case 2: layer = new Conv2D(1,1,1,1, new Function()); break;
            case 3: layer = new Pooling2D(1,1); break;
            case 4: layer = new SoftmaxLayer(1); break;
            case 5: layer = new ResBlock(ResBlock.ResConnectType_Add); break;
            case 6: layer = new Conv2DTranspose(1,1,1,1, new Function()); break;
            case 7: layer = new SelfAttention(1,1,1); break;
            case 8: layer = new FilterResponseNormalization(1); break;
            case 9: layer = new ActivationLayer(1, new Function()); break;
            case 10:layer = new Reshape(1,1,1); break;
            case 11:layer = new Dropout(0.5); break;
            case 12:layer = new SlidingWindowLayer(1, new Layer()); break;
            case 13:layer = new PositionLayer(1,1,1); break;
            case 14:layer = new CombineSequencesLayer(1); break;
            default:layer = new Layer(); break;
        }
        return layer;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder= new StringBuilder();
        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        char[] c0 = new char[32 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "(" + output_width + ", " + output_height + ", " + output_dimension  + ")";

        int v0 = 25 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');
        int param = getWeightNumber_Train();

        stringBuilder.append(name).append(c0).append(output_shape).append(c1).append(param);
        return stringBuilder.toString();
    }
}
