package haili.deeplearn.Test;

import haili.deeplearn.function.LRelu;
import haili.deeplearn.function.MSELoss;
import haili.deeplearn.function.Sigmoid;
import haili.deeplearn.model.layer.Conv2D;
import haili.deeplearn.model.layer.Dense;
import haili.deeplearn.model.Sequential;

import java.io.File;
import java.util.ArrayList;

public class Test {
    public static void main(String[] args) throws Exception {
        //test();
        int mN = 32;
        int dataNumber = mN * 9;
        ArrayList<float[]> x_train = new ArrayList<>();
        ArrayList<float[]> y_train = new ArrayList<>();

        String[] datapath = new String[1230];

//
//        File dfile_x = new File("F:\\java Project\\崩坏星穹铁道强化记录器\\data\\词条数值\\sz\\sz_data_x.txt");
//        File dfile_y = new File("F:\\java Project\\崩坏星穹铁道强化记录器\\data\\词条数值\\sz\\sz_data_y.txt");
//        dfile_x.createNewFile();
//        dfile_y.createNewFile();

        int index = 0;
        //词条识别
        File datactpath = new File("F:\\java Project\\崩坏星穹铁道强化记录器\\data\\词条数值\\sz");
        File[] files = datactpath.listFiles();
        for(File fp : files){
            String fn = fp.getName();
            if(fp.isFile()) continue;
            int label = 0;
            switch (fn){
                case "0" : label = 0; break;
                case "1" : label = 1; break;
                case "2" : label = 2; break;
                case "3" : label = 3; break;
                case "4" : label = 4; break;
                case "5" : label = 5; break;
                case "6" : label = 6; break;
                case "7" : label = 7; break;
                case "8" : label = 8; break;
                case "9" : label = 9; break;
                case "d" : label = 10; break;
                case "b" : label = 11; break;
                default: continue;
            }

            File[] images = fp.listFiles();
            for( int i = 0; i < images.length; i++){
                float[] xt =  ImageUtlis.bmpToRgbList(images[i].toString());
                x_train.add( xt );

                float[] yt = new float[12];
                yt[label] = 1.0f;
                y_train.add(yt);

                datapath[index] = images[i].toString();

//                String x = Arrays.toString(xt);
//                String y = Arrays.toString(yt);
//                x = x.substring(1, x.length()-1);
//                y = y.substring(1, y.length()-1);
//                CtModel.save_wirter(x, dfile_x.toString());
//                CtModel.save_wirter(y, dfile_y.toString());

                index++;
            }
        }

        //BpNetwork bpw = new BpNetwork("F:\\java Project\\崩坏星穹铁道强化记录器\\data\\词条数值\\sz\\bp_sz.txt");

        int error = 0;
        float[][] tx = new float[x_train.size()][];
        float[][] ty = new float[y_train.size()][];
        for(int i = 0; i < ty.length; i++){
            tx[i] = x_train.get(i);
            ty[i] = y_train.get(i);
        }


        Sequential sequential = new Sequential();
        sequential.Loss_Function = new MSELoss();
        sequential.addLayer(new Conv2D(15, 17, 3, 3, 1,  new LRelu()));
        sequential.addLayer(new Dense(13*15, 64, new LRelu()));
        sequential.addLayer(new Dense(64, 16, new LRelu()));
        sequential.addLayer(new Dense(16, ty[0].length, new Sigmoid()));

        for(int i = 0; i < ty.length; i++){

            float[] out = sequential.forward(tx[i]);
            if(getMaxIndex(ty[i]) != getMaxIndex(out)) error++;
        }
        System.out.println("error:" + error);
        System.out.println("loss:" + sequential.loss(tx, ty));

        sequential.setLearn_rate(0.01f);


        for(int i = 0; i < 10000; i++){
            for (int j = 0; j < tx.length; j++){
                sequential.backward(tx[j], ty[j]);
            }
            if(i %100 ==0) {
                System.out.println("epoch: " + i + "    loss: " + sequential.loss(tx, ty));
            }
        }


        error = 0;
        for(int i = 0; i < ty.length; i++){

            float[] out = sequential.forward(tx[i]);
            if(getMaxIndex(ty[i]) != getMaxIndex(out)) error++;
        }

        System.out.println("loss:" + sequential.loss(tx, ty));
        System.out.println("error:" + error);
    }


    static int getMaxIndex(float[] arrays){
        int index = 0;
        float max = arrays[0];
        for (int i = 0; i < arrays.length; i++){
            if(arrays[i] > max){
                max = arrays[i];
                index = i;
            }
        }
        return index;
    }
}
