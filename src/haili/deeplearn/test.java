package haili.deeplearn;

import haili.deeplearn.function.activation.Relu;

import java.util.Arrays;

public class test {
    public static void main(String[] args) {

        // input vector 输入维度
        int input_vector = 64;

        // out vector 输出维度
        int out_vector = 4;

        // 输入和隐藏层结构, 共4层，神经元个数分别是32，16，16，16，8
        int[] vars0 = new int[]{32, 16, 16, 8};

        // learn_rate 学习率
        float learn_rate = 1e-4f;

        // 创建模型，[32,16,16,8,  4]
        BpNetwork bpNetwork = new BpNetwork(input_vector, out_vector, learn_rate, new Relu(), vars0);

        // 训练
        float[][] train_x, train_y;
        train_x = new float[888][64];
        train_y = new float[888][4];
        bpNetwork.fit(train_x, train_y, 100, 64, 14);


        //输出
        float[] test_in = new float[64];
        float[] out = bpNetwork.out_(test_in);
        System.out.println("\n" +Arrays.toString(out));

        //把模型保存到文件
        bpNetwork.saveInFile("model0");

        //从文件加载模型
        BpNetwork bpNetwork2 = new BpNetwork("model0");

        //测试输出
        out = bpNetwork2.out_(test_in);
        System.out.println(Arrays.toString(out));



//        System.out.println("this out: " + Arrays.toString(out));
//        System.out.println("tensorflow out: -0.38063043, -0.31933978, -0.09566695, -0.2799532");

//
//        int spilt_N = 256;
//        int wn = 1000;
//        for (int index = 0; index < wn; index += spilt_N) {
//            int workN = spilt_N;
//            if(index + spilt_N > wn) workN = wn - index;
//            System.out.println(index + "   " + workN);
//        }
    }

}
