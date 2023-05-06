package haili.deeplearn;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.DeltaOptimizer.BaseOptimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


public class LDA {

    public float[] W;
    float[][] xE;    //每个label的特征均值
    float[] label_E; //投影后的均值
    float[] LABEL;   //所有标签
    public float loss;

    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        this.deltaOptimizer = deltaOptimizer;
    }

    BaseOptimizerInterface deltaOptimizer; //梯度优化器

    ExecutorService ThreadPool;//用于并行计算的线程池

    public LDA(int in_vector){
        W = new float[in_vector];
        loss = 9999999;
        for (int i=0; i < in_vector; i++) W[i] = 1;//Math.random() * 2 - 1;
        ThreadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        deltaOptimizer = new BaseOptimizer();
    }

    public void fit(float[][] train, float learn_rate, int epochs, int Thread_number){  fit_(train, learn_rate, epochs, Thread_number); }

    /**
     * 训练
     * @param train  train = { [x0,x1,x2,x3....], [x0,x1,x2,x3....] ,....(X0 ~ Xn-1 为特征, Xn是标签0/1) }
     * @param learn_rate 学习率
     * @param epochs 训练轮数
     */
    private void fit_(float[][] train, float learn_rate, int epochs, int Thread_number) {

        float[][][] dataSet = load_data(train);
        xE = new float[dataSet.length][W.length];
        label_E = new float[dataSet.length];


        //计算均值
        for (int i=0; i < dataSet.length; i++ ) {
            for (int j=0; j < W.length; j++ ) {
                for (int k = 0; k < dataSet[i].length; k++)
                    xE[i][j] += dataSet[i][k][j];

                xE[i][j] /= dataSet[i].length;
            }
        }

        System.out.println("开始训练:");
        float[] W_delta;
        for (int i = 0; i < epochs; i++){
            System.out.println(" epochs=" + (i+1) +"  loss=" + loss +" ...");

            if(Thread_number < 2) {
                 W_delta = WDelta(dataSet, W, 1e-10f);
                 for (int j = 0; j < W.length; j++) W[j] -=  learn_rate * deltaOptimizer.DELTA( W_delta[j], j );
            } else {
                 W_delta = WDelta( dataSet, W, 1e-10f, learn_rate, Thread_number );
                 for (int j = 0; j < W.length; j++) W[j] -=   deltaOptimizer.DELTA( W_delta[j], j );
            }
        }

        ThreadPool.shutdown();

        for (int i=0; i < dataSet.length; i++ ) {
            label_E[i] = lda( xE[i], W );
        }
    }


    /**
     * 预测
     * @param X 特征向量
     * @return 返回预测标签
     */
    public float predict(float[] X){
        float r = lda(X,W);

        int min_index = 0;
        float min_d = Math.abs(r - label_E[0]);

        for (int i=0; i < label_E.length; i++ ){
            float di = Math.abs(r - label_E[i]);
            if(di < min_d){
                min_d = di;
                min_index = i;
            }
        }
        return LABEL[min_index];
    }

    /**
     * 测试一个数据集上的准确率
     * @param test = { [x0,x1,x2,x3....], [x0,x1,x2,x3....] ,....(X0 ~ Xn-1 为特征, Xn是标签0/1) }
     * @return 准确率
     */
    public float getAccuracy(float[][] test){

        if(test[0].length != W.length + 1){
            System.out.println("test shape is " + (test[0].length-1) + ", but input_shape is " + W.length);
            return 0;
        }

        float y = 0;

        float[] Xi = new float[W.length];

        for (int i = 0; i < test.length; i++){
            System.arraycopy(test[i], 0, Xi, 0, Xi.length);
            if(test[i][W.length] == predict(Xi)) y++;
        }

        y /= test.length;

        return y;
    }


    /**
     * 数据处理
     * @param train ..
     * @return { label1[][], abel2[][], abel3[][], .... }
     */
    private float[][][] load_data(float[][] train){
        //key:label, v:train float[][]
        Map<String, ArrayList<float[]> > train_label_map = new HashMap();

        for(int i=0; i < train.length; i++){

                int label_index = train[i].length - 1;

                String label = train[i][ label_index ] + "";

                float[] Xi = new float[label_index];
                System.arraycopy(train[i], 0, Xi, 0, label_index);

                if(train_label_map.containsKey(label)){
                    train_label_map.get(label).add(Xi);
                } else {
                    ArrayList<float[]> arrayList = new ArrayList<>();
                    arrayList.add(Xi);
                    train_label_map.put(label,arrayList);
                }

        }

        float[][][] train_label = new float[train_label_map.size()][][];
        LABEL = new float[train_label_map.size()];


        int index = 0;
        for (String key: train_label_map.keySet()) {

                ArrayList<float[]> list =  train_label_map.get(key);
                float[][] li = new float[list.size()][];

                for(int j = 0; j < li.length; j++) li[j] = list.get(j);

                train_label[index] = li;

                LABEL[index] = Float.parseFloat(key);
                //System.out.println(key);
                index++;
        }

        return train_label;
    }


    /**
     * fisher投影
     * @param X 特征
     * @param W  w
     * @return 值
     */
    private float lda(float[] X, float[] W){
        float r = 0;
        for(int i=0; i < W.length; i++) r += W[i]*X[i];
        return r;
    }


    /**
     * 目标优化函数，越小越好
     * @param train_label { label1[][], abel2[][], abel3[][], .... }
     * @param w W
     * @return
     */
    private float loss(float[][][] train_label, float[] w){
        //Y,每个特征投影后的值
        float[][] Y = new float[train_label.length][];

        //每个label均值
        float[] yE = new float[train_label.length];

        //每个label方差
        float[] yD = new float[train_label.length];


        for(int i=0; i < train_label.length; i++){

            float[] yi = new float[train_label[i].length];

            for (int j=0; j < yi.length; j++)
                yi[j] = lda( train_label[i][j], w);

            Y[i] = yi;

            yE[i] = lda(xE[i], w);
            yD[i] = D( Y[i] );
        }

        //计算损失
        loss = loss_function( yE, yD );

        return loss;
    }



    /**
     * 计算损失函数
     * @param yE 均值
     * @param yD 方差
     * @return loss
     */
    public float loss_function(float[] yE, float[] yD){
        float d = 0;
        for(float di: yD) d += di;

        float mse = 0;

        for(int i=0; i < yE.length-1; i++){
            for (int j = i+1; j < yE.length; j++){
                float a = yE[i] - yE[j];
                a = a * a;
                mse += a;
            }
        }

        d = d / mse;

        return d;
    }


    /**
     * 计算数组均值
     * @param x X
     * @return E
     */
    private float E(float[] x){
        float r = 0;
        for (float xi : x) r += xi;
        r /= x.length;
        return r;
    }


    /**
     * 计算方差
     * @param x x
     * @return d
     */
    private float D(float[] x){
        float r = 0;
        float E = E(x);
        for(float xi : x){
            float d = xi - E;
            r += d * d;
        }
        return r;
    }


    /**
     * 计算梯度
     * @param train_label 。
     * @param x 。
     * @param e 精度
     * @return x的梯度
     */
    private  float[] WDelta(float[][][] train_label, float[] x, float e){
        float[] r = new float[x.length];

        float var0 = loss(train_label, x);
        for(int i=0; i<r.length; i++){
            float[] nx = x.clone();
            nx[i] += e;
            r[i] = (loss(train_label, nx) - var0) / e;
        }

        return r;
    }

    private  float[] WDelta(float[][][] train_label, float[] x, float e, float learn_rate, int Thread_number) {
        DeltaWThreadPool upthread = new DeltaWThreadPool(train_label, x, e, learn_rate);
        Future[] futureList = new Future[Thread_number];
        for(int i = 0; i < Thread_number; i++ )
            futureList[i] = ThreadPool.submit(new Thread(upthread));
        try {
            for(Future future: futureList) future.get();
        }catch (InterruptedException | ExecutionException exception){
            exception.printStackTrace();
        }
        return upthread.getWDelta();
    }

    private class DeltaWThreadPool implements Runnable{

        final private float[][][] train_label;
        final private float var0;
        final private float e;
        final private float learn_rate;
        final private float[] tW;
        private float[] WDelta;
        private boolean[] is_finish;
        private boolean[] is_lock;
        private boolean finish;


        public DeltaWThreadPool(float[][][] train_label, float[] w, float e, float learn_rate){
            finish = false;
            WDelta = new float[W.length];
            is_finish = new boolean[W.length];
            is_lock = new boolean[W.length];
            tW = w;
            this.e = e;
            this.learn_rate = learn_rate;
            this.train_label = train_label;
            var0 = loss(train_label, w);

        }

        @Override
        public void run() {
            while (!finish) {
                boolean flag = true;
                for (int i = 0; i < WDelta.length; i++)
                    if (!is_finish[i] && !is_lock[i]) {
                        flag = false;
                        is_lock[i] = true;

                        float[] nx = tW.clone();
                        nx[i] += e;
                        WDelta[i] = learn_rate * (  (loss(train_label, nx) - var0) / e  );

                        is_finish[i] = true;
                    }
                finish = flag;
            }
        }//End run

        public float[] getWDelta(){
            return WDelta;
        }
    }

}

