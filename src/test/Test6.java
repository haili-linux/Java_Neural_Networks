package test;

import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.Sigmoid;
import haili.deeplearn.function.activation.Tanh;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;

public class Test6 {
    public static void main(String[] args) {
        int dem = 128;
        Sequential sr0 = new Sequential(dem);
        sr0.addLayer(new Dense(dem, 1, dem, 1, dem, new Function(), false));
        sr0.addLayer(new FilterResponseNormalization());
        sr0.addLayer(new ActivationLayer(new Tanh()));

        Sequential sr01 = new Sequential(dem);
        sr01.addLayer(new Dense(dem, 1, dem, 1, dem, new Function(), false));
        sr01.addLayer(new FilterResponseNormalization());
        sr01.addLayer(new ActivationLayer(new Tanh()));

        Sequential sr02 = new Sequential(dem);
        sr02.addLayer(new Dense(dem, 1, dem, 1, dem, new Function(), false));
        sr02.addLayer(new FilterResponseNormalization());
        sr02.addLayer(new ActivationLayer(new Tanh()));

        ResBlock resBlock0 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock0.addLayer(new SelfAttention(dem, dem, 64));
        //resBlock0.addLayer(new SlidingWindowLayer(dem, sr0));

        ResBlock resBlock1 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock1.addLayer(new SelfAttention(dem, dem, 64));
        //resBlock1.addLayer(new SlidingWindowLayer(dem, sr01));

        ResBlock resBlock2 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock2.addLayer(new SelfAttention(dem, dem, 64));
        //resBlock2.addLayer(new SlidingWindowLayer(dem, sr02));

        ResBlock resBlock3 = new ResBlock(ResBlock.ResConnectType_Add);
        resBlock3.addLayer(new SelfAttention(dem, dem, 64));
        resBlock3.addLayer(new SlidingWindowLayer(dem, new Dense(dem, new Tanh())));

        //input= { 胚子, 强化结果n, 强化结果n-1, 强化结果n-2, ..., 强化结果0 }
        Sequential sequential = new Sequential();
        sequential.addLayer(new SlidingWindowLayer(67, new Dense(dem - 1, new Tanh())));
        sequential.addLayer(new PositionLayer(dem - 1, 1, 24));

        sequential.addLayer(resBlock0);
        sequential.addLayer(resBlock1);
        //sequential.addLayer(resBlock2);
        //sequential.addLayer(resBlock3);

        sequential.addLayer(new CombineSequencesLayer(dem));
        sequential.addLayer(new FilterResponseNormalization());
        sequential.addLayer(new ActivationLayer(new Tanh()));

        sequential.addLayer(new Dense(14, new Sigmoid()));

        float[][] x = new float[][]{new float[67*15]};
        float[][] y = new float[][]{new float[14]};
        sequential.backward(new float[67*15], new float[14], new float[14]);
    }
}
