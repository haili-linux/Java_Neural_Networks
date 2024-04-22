package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizer;
import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.softmax.SoftmaxLayer;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

public class Layer implements LayerInterface{

    public int id = -1;

    public float learn_rate = 1e-4f;

    public int input_dimension = 0;
    public int input_width = 0, input_height = 0;

    public int output_dimension = 0;
    public int output_width = 0, output_height = 0;

    public Function activation_function = new Function();

    public boolean use_bias = true;

    protected BaseOptimizerInterface deltaOptimizer = new BaseOptimizer();

    //是否保存中间变量
    public boolean saveHiddenLayerOutput = false;

    //用于保存中间变量
    public Map<float[], Object> hiddenLayerOutputMap = new HashMap<>();

    @Override
    public void init(int input_width, int input_height, int input_Dimension) { }

    @Override
    public float[] forward(float[] inputs) {
        return new float[0];
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
        return new float[0][];
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) { }

    //返回参数个数
    public int getWeightNumber(){ return 0; }

    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        this.deltaOptimizer = deltaOptimizer;
    }

    public void setActivation_Function(Function activation){
        this.activation_function = activation;
    }

    public void initByFile(BufferedReader in) throws Exception{

    }

    public void saveInFile(PrintWriter pw) throws Exception{ }

    public void setLearn_rate(float learn_rate){
        this.learn_rate = learn_rate;
    }

    public void setSaveHiddenLayerOutput(boolean b){
        this.saveHiddenLayerOutput = b;
    }

    public void clearHiddenLayerOutput(){
        this.hiddenLayerOutputMap.clear();
    }

    public final static Layer getLayerById(int id){
        Layer layer;
        switch (id){
            case 0: layer = new Sequential(-1, -1, -1); break;
            case 1: layer = new Dense(1, new Function()); break;
            case 2: layer = new Conv2D(1,1,1,1, new Function()); break;
            case 3: layer = new Pooling2D(1,1); break;
            case 4: layer = new SoftmaxLayer(1); break;
            case 5: layer = new ResBlock(ResBlock.ResConnectType_Add); break;
            case 6: layer = new Conv2DTranspose(1,1,1,1, new Function()); break;
            case 7: layer = new SelfAttention(1,1,1); break;
            case 8: layer = new FilterResponseNormalization(1); break;
            case 9: layer = new ActivationLayer(1, new Function()); break;
            default: layer = new Layer(); break;
        }
        return layer;
    }

}
