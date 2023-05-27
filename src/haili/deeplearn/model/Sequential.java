package haili.deeplearn.model;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Fuction;
import haili.deeplearn.function.MSELoss;
import haili.deeplearn.model.layer.Conv2D;
import haili.deeplearn.model.layer.Dense;
import haili.deeplearn.model.layer.Layer;
import haili.deeplearn.model.layer.Pooling2D;
import haili.deeplearn.utils.DataSetUtils;
import haili.deeplearn.utils.ProgressBarCmd;
import haili.deeplearn.utils.SaveData;
import haili.deeplearn.utils.ThreadWork;

import java.io.*;
import java.util.ArrayList;


public class Sequential extends Layer{

    public String EXPLAIN = "";

    public Fuction Loss_Function = new MSELoss();

    public float loss = 0;

    private float learn_rate = 1e-4f;

    public ArrayList<Layer> layers = new ArrayList<>();


    public Sequential(int input_width, int input_height, int input_Dimension){
        id = 0;
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_Dimension;
    }

    public Sequential(){ id = 0; }

    /**
     * 通过模型文件初始化
     * @param fileName 文件路径
     */
    public Sequential(String fileName){
        id = 0;
        try {
            initByFile(fileName);
        } catch (Exception exception) {
            exception.printStackTrace();
        }
    }


    public void addLayer(Layer layer){

        if(layer==this){
            System.out.println(" error!");
            return;
        }

        layer.learn_rate = learn_rate;

        //layer初始化
        if(layer.input_dimension == 0 || layer.input_height == 0 || layer.input_width == 0){
            if(layers.size() == 0){
                if(input_width!=0 && input_height!=0 && input_dimension!=0)
                    layer.init(input_width, input_height, input_dimension);
            } else {
                Layer lastLayer = layers.get(layers.size()-1);
                layer.init(lastLayer.output_width, lastLayer.output_height, lastLayer.output_dimension);
            }
        }

        layers.add(layer);
        output_dimension = layer.output_dimension;
        output_width = layer.output_width;
        output_height = layer.output_height;
    }

    public void setLoss_Function(Fuction loss_Function){
        this.Loss_Function = loss_Function;
    }
    public void setLearn_rate(float learn_rate){
        this.learn_rate = learn_rate;
        for (Layer layer: layers){
            layer.learn_rate = learn_rate;
        }
    }
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer){
        for (Layer layer: layers){
            layer.setDeltaOptimizer(deltaOptimizer);
        }
    }

    /**
     *  对模型进行训练
     * @param train_X x
     * @param train_Y  y
     * @param batch_size bs
     * @param epoch epoch
     * @param Thread_n 使用的cpu线程数量
     * @return  训练完成后的在x，y上的loss值
     */
    public float fit(float[][] train_X, float[][] train_Y, int batch_size, int epoch, int Thread_n){
        //参数检查
        if(batch_size<1 &&epoch <0) return 99999;
        //获取cpu核心数
        int core_number = Runtime.getRuntime().availableProcessors();
        if(Thread_n>core_number) Thread_n = core_number;

        if(batch_size == 1) {
            for(int i = 0; i < epoch; i++){

                if(i%100==0) System.out.println( "  epoch: " + (i + 1) + "  " + calculateLoss(train_X, train_Y) );

                for (int j = 0; j < train_X.length; j++){
                    float[][] d = backward(train_X[j], train_Y[j]);
                    upgradeWeight(d);
                }
            }
            System.out.println("");
        } else if(batch_size >= train_X.length) {//batch_size和训练集一样，全批量梯度下降
            for (int i = 0; i < epoch; i++) {
                upgradeBatch(train_X, train_Y, Thread_n);
            }
        } else {//mini-batch

            ArrayList<float[][]>[] data = DataSetUtils.splitBatch(train_X, train_Y, batch_size);
            ArrayList<float[][]> train_x = data[0];
            ArrayList<float[][]> train_y = data[1];

            for (int i = 0; i < epoch; i++){
                String title = "  epoch: " + (i + 1) + "  ";
                upgrade_mini_batch_progressbar(Thread_n, train_x, train_y, title);
            }
            System.out.println("");
        }

        return loss = calculateLoss(train_X, train_Y);
    }

    private void upgrade_mini_batch_progressbar(int Thread_n, ArrayList<float[][]> train_x, ArrayList<float[][]> train_y, String title) {
        ProgressBarCmd progressBarCmd = new ProgressBarCmd(title, train_x.size(), 50);
        System.out.print(progressBarCmd.setProgress(0));
        for (int i = 0; i < train_x.size(); i++) {
            upgradeBatch(train_x.get(i), train_y.get(i), Thread_n);
            System.out.print(progressBarCmd.setProgress(i + 1));
        }
    }

    private void upgradeBatch(float[][] train_X, float[][] train_Y, int threadNumber){
        float[][][] deltas = new float[train_X.length][][];
        ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(train_X.length){
            @Override
            public void working(int index) {
                deltas[index] = backward(train_X[index], train_Y[index]);

            }
        };
        ThreadWork.start(threadWorker, threadNumber);


        for(int i = 0; i < deltas[0].length; i++) {

            if(deltas[0][i]==null) continue;

            int n = deltas[0][i].length;
            final int layerIndex = i;
            ThreadWork.ThreadWorker threadWorker2 = new ThreadWork.ThreadWorker(n) {
                @Override
                public void working(int index) {
                    for (int i = 0; i < deltas.length; i++){
                        deltas[0][layerIndex][index] += deltas[i][layerIndex][index];
                    }
                    deltas[0][layerIndex][index] /= deltas.length;
                }
            };

            threadWorker2.setStart_index(1);
            ThreadWork.start(threadWorker2, threadNumber);
        }

        upgradeWeight(deltas[0]);
    }


    /**
     * output
     * @param inputs inputs
     * @return 网络每层的的输出
     */
    private ArrayList<float[]> forward_list(float[] inputs){
        ArrayList<float[]> output = new ArrayList<>();
        output.add( layers.get(0).forward(inputs) );

        for(int i = 1; i < layers.size(); i++){
            output.add( layers.get(i).forward(output.get(i-1)) );
        }
        return output;
    }


    /**
     * backward 计算梯度
     * @param x_train x
     * @param y_train y
     * @return 每层的参数梯度
     */
    public float[][] backward(float[] x_train, float[] y_train){
        ArrayList<float[]> output = forward_list(x_train);

        float[][] w_deltas = new float[layers.size()][];
        float[][] deltas = new float[2][];

        deltas[0] = lossDelta(output.get(output.size()-1), y_train);

        for(int i = output.size()-1; i > 0; i--) {
            deltas = layers.get(i).backward(output.get(i - 1), output.get(i), deltas[0]);
            w_deltas[i] = deltas[1];
        }

        deltas = layers.get(0).backward(x_train, output.get(0), deltas[0]);
        w_deltas[0] = deltas[1];

        return w_deltas;
    }


    /**
     * 更新梯度
     * @param w_deltas backward()返回的梯度
     */
    public void upgradeWeight(float[][] w_deltas){
        for(int i = layers.size()-1; i > 0; i--) {
            //System.out.println("layer: " + i);
            layers.get(i).upgradeWeight(w_deltas[i]);
        }
    }


    //测试一个数据集上的误差
    public float calculateLoss(float[][] train_X, float[][] train_Y){

        ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(train_X.length){

            final float[] loss = new float[train_X.length];

            @Override
            public void working(int index) {
                loss[index] = loss(train_X[index], train_Y[index]);
            }

            @Override
            public Object getObject() {
                return loss;
            }
        };

        ThreadWork.start(threadWorker);

        float error = 0;
        for (float loss : (float[]) threadWorker.getObject()){
            error += loss;
        }

        return loss = error / train_X.length;

    }



    //计算loss层梯度
    private float[] lossDelta(float[] out, float[] y_train){
        float[] delta = new float[y_train.length];
        for(int i = 0; i < y_train.length; i++){
            delta[i] = Loss_Function.f_derivative(out[i], y_train[i]);
        }
        return delta;
    }
    private float loss(float[] x, float[] y){
        float loss = 0;
        float[] oi = forward(x);
        for (int j = 0; j < oi.length; j ++){
            loss += Loss_Function.f(oi[j], y[j]);
        }
        return loss;
    }


    /**
     * 保存模型
     * @param fileName 文件路径
     * @throws Exception IOE
     */
    public void saveInFile(String fileName) throws Exception{
        File file = new File(fileName);
        if (file.exists()) {
            String p1 = fileName.substring(0, fileName.lastIndexOf("."));
            for (int i = 0; ; i++) {
                file = new File(fileName = p1 + "_" + i + ".log");
                if (!file.exists()) break;
            }
        }
        try { boolean newFile = file.createNewFile(); } catch (Exception ignored) { }

        if(file.isFile()) {
            FileWriter fw = null;
            try { fw = new FileWriter(file, true); } catch (IOException e) { e.printStackTrace(); return; }

            PrintWriter pw = new PrintWriter(fw);

            saveInFile(pw);

            pw.flush();
            try {
                fw.flush();
                pw.close();
                fw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void initByFile(String fileName) throws Exception{
        File file = new File(fileName);
        if(file.isFile())
            try {
                FileReader fileReader = null;
                fileReader = new FileReader(file);
                BufferedReader in = new BufferedReader(fileReader);

                in.readLine();
                //System.out.println(in.readLine());
                initByFile(in);

                in.close();
                fileReader.close();
            }catch(Exception e){
                e.printStackTrace();
            }
    }

    private Layer getLayerById(int id){
        Layer layer;
        switch (id){
            case 0: layer = new Sequential(-1, -1, -1); break;
            case 1: layer = new Dense(1, new Fuction()); break;
            case 2: layer = new Conv2D(1,1,1,1, new Fuction()); break;
            case 3: layer = new Pooling2D(1,1); break;
            default: layer = new Layer(); break;
        }
        return layer;
    }

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        if(layers.size()>0){
            layers.get(0).init(input_width, input_height, input_Dimension);
        }
    }

    @Override
    public float[] forward(float[] inputs){
        float[] out = layers.get(0).forward(inputs);
        for(int i = 1; i < layers.size(); i++){
            out = layers.get(i).forward(out);
        }
        return out;
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        ArrayList<float[]> output_list = forward_list(inputs);

        float[] w_deltas = new float[getWeightNumber()];
        int index = 0;

        float[][] back = new float[2][];
        back[0] = deltas;
        for(int i = output_list.size()-1; i > 0; i--) {
            back = layers.get(i).backward(output_list.get(i - 1), output_list.get(i), back[0]);
            System.arraycopy(back[1], 0, w_deltas, index, back[1].length);
            index += back[1].length;
        }

        back = layers.get(0).backward(inputs, output_list.get(0), back[0]);
        System.arraycopy(back[1], 0, w_deltas, index, back[1].length);

        return new float[][]{back[0], w_deltas};
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) {
        int index = 0;
        for(int i = layers.size()-1; i >= 0; i--) {
            float[] w_delta = new float[layers.get(i).getWeightNumber()];
            if (w_delta.length >= 0) System.arraycopy(weightDeltas, index, w_delta, 0, w_delta.length);
            index += w_delta.length;
            layers.get(i).upgradeWeight(w_delta);
        }
    }

    @Override
    public int getWeightNumber() {
        int n = 0;
        for (Layer layer : layers){
            n += layer.getWeightNumber();
        }
        return n;
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println("explain: Sequential:" + EXPLAIN);

        pw.println(SaveData.sInt("input_Dimension", input_dimension));
        pw.println(SaveData.sInt("input_width", input_width));
        pw.println(SaveData.sInt("input_height", input_height));

        pw.println(SaveData.sInt("output_Dimension", output_dimension));
        pw.println(SaveData.sInt("output_width", output_width));
        pw.println(SaveData.sInt("output_height", output_height));


        pw.println(SaveData.sFloat("loss", loss));
        pw.println(SaveData.sFloat("learn_rate", learn_rate));
        pw.println(SaveData.sInt("LossFunction", Loss_Function.id));

        for (Layer layer : layers)
            layer.saveInFile(pw);
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        String line = in.readLine();
        EXPLAIN = line.substring(20);

        input_dimension = SaveData.getSInt(in.readLine());
        input_width = SaveData.getSInt(in.readLine());
        input_height = SaveData.getSInt(in.readLine());

        output_dimension = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());
        output_height = SaveData.getSInt(in.readLine());

        loss = SaveData.getSFloat(in.readLine());
        learn_rate = SaveData.getSFloat(in.readLine());
        Loss_Function = Fuction.getFunctionById(SaveData.getSInt(in.readLine()));

        while ((line=in.readLine()) != null){
            Layer layer = getLayerById(SaveData.getSInt(line));
            layer.initByFile(in);
            layers.add(layer);
        }

        setLearn_rate(learn_rate);
    }

    @Override
    public String toString() {
        return "Sequential{" +
                "input_width=" + input_width +
                ", input_height=" + input_height +
                ", input_Dimension=" + input_dimension +
                ", Loss_Function=" + Loss_Function +
                ", learn_rate=" + learn_rate +
                ", layers=" + layers +
                '}';
    }
}
