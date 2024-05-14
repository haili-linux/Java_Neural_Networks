package test;

import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Sigmoid;
import haili.deeplearn.function.activation.Tanh;
import haili.deeplearn.function.loss.CELoss;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;

public class Test6 {
    public static void main(String[] args) {

        String modelName = "C:\\Tool\\IDEA\\Java Project\\原神圣遗物强化分析\\art分析3\\model\\attention_model__b_3.txt";

        int dem = 128;
        Sequential sr0 = new Sequential(dem);
        sr0.addLayer(new Dense(dem, 1, dem, 1, dem, new Function(), false));
        sr0.addLayer(new FilterResponseNormalization());
        sr0.addLayer(new ActivationLayer(new LRelu()));

        Sequential sr01 = new Sequential(dem);
        sr01.addLayer(new Dense(dem, 1, dem, 1, dem, new Function(), false));
        sr01.addLayer(new FilterResponseNormalization());
        sr01.addLayer(new ActivationLayer(new LRelu()));

        Sequential sr02 = new Sequential(dem);
        sr02.addLayer(new Dense(dem, 1, dem, 1, dem, new Function(), false));
        sr02.addLayer(new FilterResponseNormalization());
        sr02.addLayer(new ActivationLayer(new LRelu()));

        ResBlock resBlock0 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock0.addLayer(new SelfAttention(dem, dem, 64));
        resBlock0.addLayer(new SlidingWindowLayer(dem, sr0));

        ResBlock resBlock1 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock1.addLayer(new SelfAttention(dem, dem, 64));
        resBlock1.addLayer(new SlidingWindowLayer(dem, sr01));

        ResBlock resBlock2 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock2.addLayer(new SelfAttention(dem, dem, 64));
        resBlock2.addLayer(new SlidingWindowLayer(dem, sr02));

        ResBlock resBlock3 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock3.addLayer(new SelfAttention(dem, dem, 64));
        resBlock3.addLayer(new SlidingWindowLayer(dem, new Dense(dem, new LRelu())));



        ResBlock resBlock_inputs = new ResBlock(ResBlock.ResConnectType_Concat);
        resBlock_inputs.addLayer(new SlidingWindowLayer(67, new Dense(dem - 1, new LRelu())));
        resBlock_inputs.addLayer(new PositionLayer(dem - 1, 1, 24));
        resBlock_inputs.addLayer(resBlock0);
        resBlock_inputs.addLayer(resBlock1);
        resBlock_inputs.addLayer(new CombineSequencesLayer(new Dense(dem, 1, dem * 2, 1,dem * 2, new Function(), false)));
        resBlock_inputs.addLayer(new FilterResponseNormalization());
        resBlock_inputs.addLayer(new ActivationLayer(new LRelu()));

        //input= { 胚子, 强化结果n, 强化结果n-1, 强化结果n-2, ..., 强化结果0 }
        Sequential sequential = new Sequential();
        sequential.addLayer(resBlock_inputs);
        sequential.addLayer(new Dense(dem * 2 + 50, 1, 64, 1, 64, new Function(), false));
        sequential.addLayer(new FilterResponseNormalization());
        sequential.addLayer(new ActivationLayer(new LRelu()));
        sequential.addLayer(new Dense(14, new Sigmoid()));

        //sequential.setSaveHiddenLayerOutput(true);
        sequential = new Sequential( modelName);

        System.out.println(sequential.summary());

        sequential.setLoss_Function(new CELoss());
        sequential.setDeltaOptimizer(new Adam());

        float[][] x = new float[][]{new float[67*15]};
        float[][] y = new float[][]{new float[14]};
        sequential.backward(new float[67*15], new float[14], new float[14]);
    }
}
