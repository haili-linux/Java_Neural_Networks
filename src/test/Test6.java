package test;

import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Sigmoid;
import haili.deeplearn.function.activation.Tanh;
import haili.deeplearn.function.loss.CELoss;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;

import java.util.Arrays;

public class Test6 {
    public static void main(String[] args) {
        SplitLayer splitLayer = new SplitLayer(2, 2);
        float[] inputs = {1, 2, 3, 4, 5};
        float[] outputs = splitLayer.forward(inputs);
        System.out.println(Arrays.toString(outputs));
    }
}
