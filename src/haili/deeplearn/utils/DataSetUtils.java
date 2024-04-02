package haili.deeplearn.utils;

import haili.deeplearn.BpNetwork;
import org.bouncycastle.jce.provider.PEMUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class DataSetUtils {



    public static ArrayList<float[][]>[]  splitBatch(float[][] in, float[][] t, int batch_size){
        TrainData[] trainDatas = new TrainData[in.length];
        for (int i = 0; i < trainDatas.length; i++)
            trainDatas[i] = new TrainData(in[i], t[i]);

        ArrayList<TrainData> d0 = new ArrayList<>();
        Collections.addAll(d0, trainDatas);

        //随机打乱
        Collections.shuffle(d0);
        //System.out.println("splitBatch(): d0.size=" + d0.size() + "   " + d0.get(in.length - 1));

        ArrayList<TrainData[]> trainDataList = new ArrayList<>();

        TrainData[] var0 = new TrainData[batch_size];
        for (int i = 0; i < d0.size(); i++){
            TrainData ti = d0.get(i);
            if( i%batch_size == 0 && i>0){
                trainDataList.add(var0);

                int n = d0.size() - i;
                if(n > batch_size) n = batch_size;
                var0 = new TrainData[n];
            }
            var0[i%batch_size] = ti;
        }

        //System.out.println("splitBatch(): trainDataList.size=" + trainDataList.size());


        if(in.length % batch_size > 16){
            trainDataList.add(var0);
            //System.out.println(trainDataList.get(trainDataList.size()-1).length + "   " + in.length % batch_size);
        }

        ArrayList<float[][]> train_x, train_y;
        train_x = new ArrayList<>();
        train_y = new ArrayList<>();
        for (TrainData[] di : trainDataList){
            //System.out.println(di);
            float[][] x = new float[di.length][];
            float[][] y = new float[di.length][];
            for(int i = 0; i < di.length; i++){

                x[i] = di[i].train_x;
                y[i] = di[i].train_y;
            }
            train_x.add(x);
            train_y.add(y);
        }

        //System.out.println("splitBatch(): train_x: " + train_x.size() + "    " + Arrays.toString(train_x.get(0)));
        //System.out.println("splitBatch(): train_y: " + train_y.size() + "    " + Arrays.toString(train_y.get(0)));

        ArrayList[] data = new ArrayList[2];
        data[0] = train_x;
        data[1] = train_y;
        return data;
    }


}

