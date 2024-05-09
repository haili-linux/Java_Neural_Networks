package haili.deeplearn.model;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.Softmax;
import haili.deeplearn.model.layer.Dense;
import haili.deeplearn.model.layer.Layer;
import haili.deeplearn.model.layer.SoftmaxLayer;
import haili.deeplearn.model.loss.LossLayer;
import haili.deeplearn.utils.DataSetUtils;
import haili.deeplearn.utils.ProgressBarCmd;
import haili.deeplearn.utils.SaveData;
import haili.deeplearn.utils.ThreadWork;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;


public class Sequential extends Layer{

    public String EXPLAIN = "";

    public float loss = 0;

    private float learn_rate = 1e-4f;

    public ArrayList<Layer> layers = new ArrayList<>();

    public LossLayer lossLayer = new LossLayer();

    public Sequential(int input_width, int input_height, int input_Dimension){
        id = 0;
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_dimension = input_Dimension;
    }

    public Sequential(int input_Dimension){
        id = 0;
        this.input_width = input_Dimension;
        this.input_height = 1;
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

        if(layer == this){
            System.out.println(" error!");
            System.exit(0);
        }

        layer.learn_rate = learn_rate;

        //layer初始化
        if(layer.input_dimension == 1 || layer.input_height == 1 || layer.input_width == 1){
            if(layers.isEmpty()){
                if( (input_width>0 && input_height>0) || input_dimension>0)
                    layer.init(input_width, input_height, input_dimension);
            } else {
                Layer lastLayer = layers.get(layers.size()-1);
                layer.init(lastLayer.output_width, lastLayer.output_height, lastLayer.output_dimension);
            }
        }


        layers.add(layer);

        //是全连接层
        if(layer.id == new Dense(1, new Function()).id) {
            if (layer.activation_function.id == new Softmax().id)//激活函数是softmax
                layers.add(new SoftmaxLayer(layer.output_dimension));
        }

        output_dimension = layer.output_dimension;
        output_width = layer.output_width;
        output_height = layer.output_height;
    }

  
    public void setLoss_Function(Function loss_Function){
        this.lossLayer.loss_function = loss_Function;
    }

    @Override
    public void setLearn_rate(float learn_rate){
        this.learn_rate = learn_rate;
        for (Layer layer: layers){
            layer.setLearn_rate(learn_rate);
        }
    }

    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer){
        for (Layer layer: layers){
            layer.setDeltaOptimizer(deltaOptimizer);
        }
    }

    @Override
    public void setTrain(boolean train) {
        this.train = train;
        for (Layer layer: layers){
            layer.setTrain(this.train);
        }
    }

    /**
     * 对模型进行训练
     *
     * @param train_X    x
     * @param train_Y    y
     * @param batch_size bs
     * @param epoch      epoch
     * @param Thread_n   使用的cpu线程数量
     */
    public void fit(float[][] train_X, float[][] train_Y, int batch_size, int epoch, int Thread_n){
        //参数检查
        if(batch_size<1 &&epoch <0) return;
        //获取cpu核心数
        int core_number = Runtime.getRuntime().availableProcessors();
        if(Thread_n>core_number) Thread_n = core_number;

        setTrain(true);
        if(batch_size == 1) {
            for(int i = 0; i < epoch; i++){

                if(i%100==0) System.out.println( "  epoch: " + (i + 1) + "  " + calculateLoss(train_X, train_Y) );

                for (int j = 0; j < train_X.length; j++){
                    float[][] d = backward(train_X[j], train_Y[j]);
                    upgradeWeight(d);
                }
            }
            System.out.println();
        } else if(batch_size >= train_X.length) {//batch_size和训练集一样，全批量梯度下降
            for (int i = 0; i < epoch; i++) {
                upgradeBatch(train_X, train_Y, Thread_n);
            }
        } else {//mini-batch
            for (int i = 0; i < epoch; i++){
                ArrayList<float[][]>[] data = DataSetUtils.splitBatch(train_X, train_Y, batch_size);
                ArrayList<float[][]> train_x = data[0];
                ArrayList<float[][]> train_y = data[1];

                String title = "  epoch: " + (i + 1) + "  ";
                upgrade_mini_batch_progressbar(Thread_n, train_x, train_y, title);
            }
            System.out.println();
        }

        setTrain(false);
    }

    private void upgrade_mini_batch_progressbar(int Thread_n, ArrayList<float[][]> train_x, ArrayList<float[][]> train_y, String title) {
        ProgressBarCmd progressBarCmd = new ProgressBarCmd(title, train_x.size(), 50);
        System.out.print(progressBarCmd.setProgress(0));
        for (int i = 0; i < train_x.size(); i++) {
            upgradeBatch(train_x.get(i), train_y.get(i), Thread_n);
            System.out.print(progressBarCmd.setProgress(i + 1));
        }
    }

    public void setSaveHiddenLayerOutput(boolean b){
        this.saveHiddenLayerOutput = b;
        for (Layer layer: layers)
            layer.setSaveHiddenLayerOutput(b);
    }

    public void clearHiddenLayerOutput(){
        this.hiddenLayerOutputMap.clear();
        for (Layer layer: layers)
            layer.clearHiddenLayerOutput();
    }


    public boolean SaveHiddenLayerOutput = false;

    /**
     * @param threadNumber 使用的线程数量
     * @return 模型的参数的梯度
     */
    public float[][] gradient(float[][] train_X, float[][] train_Y, int threadNumber) {

        if(SaveHiddenLayerOutput)
            setSaveHiddenLayerOutput(SaveHiddenLayerOutput);

        float[][] deltas_layer;
        if(train_X.length > 1 && threadNumber > 1) {
            // bach中每个的梯度
            float[][][] deltas = new float[train_X.length][][];
            ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(train_X.length) {
                @Override
                public void working(int index) {
                    deltas[index] = backward(train_X[index], train_Y[index]);

                }
            };
            ThreadWork.start(threadWorker, threadNumber);

            // 求bach的梯度均值
            for (int i = 0; i < deltas[0].length; i++) {

                if (deltas[0][i] == null) continue;

                int n = deltas[0][i].length;
                final int layerIndex = i;
                ThreadWork.ThreadWorker threadWorker2 = new ThreadWork.ThreadWorker(n) {
                    @Override
                    public void working(int index) {
                        for (int i = 0; i < deltas.length; i++) {
                            deltas[0][layerIndex][index] += deltas[i][layerIndex][index];
                        }
                        deltas[0][layerIndex][index] /= deltas.length;
                    }
                };

                threadWorker2.setStart_index(1);
                ThreadWork.start(threadWorker2, threadNumber);
            }

            deltas_layer = deltas[0];

        } else {
            deltas_layer = backward(train_X[0], train_Y[0]);
        }

        if (SaveHiddenLayerOutput) {
            // 清楚中间变量缓存
            setSaveHiddenLayerOutput(false);
            clearHiddenLayerOutput();
        }

        return deltas_layer;
    }

    private void upgradeBatch(float[][] train_X, float[][] train_Y, int threadNumber){
        //计算参数梯度
        float[][] deltas_layer = gradient(train_X, train_Y,  threadNumber);
        //使用梯度均值更新模型权重
        upgradeWeight(deltas_layer);
    }


    /**
     * output
     * @param inputs inputs
     * @return 网络每层的的输出
     */
    public ArrayList<float[]> forward_list(float[] inputs){
        ArrayList<float[]> output;
        if(saveHiddenLayerOutput && hiddenLayerOutputMap.containsKey(inputs)) {
            output = (ArrayList<float[]>) hiddenLayerOutputMap.get(inputs);
            if(output != null)
                return output;
        }

        output = new ArrayList<>();
        output.add( layers.get(0).forward(inputs) );

        for(int i = 1; i < layers.size(); i++){
            output.add( layers.get(i).forward(output.get(i-1)) );
        }

        // 保存中间输出
        if(saveHiddenLayerOutput){
            hiddenLayerOutputMap.put(inputs, output);
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

        deltas[0] = lossLayer.gradient(output.get(output.size()-1), y_train);

        for(int i = output.size()-1; i > 0; i--) {
            deltas = layers.get(i).backward(output.get(i - 1), output.get(i), deltas[0]);
            w_deltas[i] = deltas[1];
        }

        deltas = layers.get(0).backward(x_train, output.get(0), deltas[0]);
        w_deltas[0] = deltas[1];

        return w_deltas;
    }


    /**
     * 更新权重梯度
     * @param w_deltas gradient()返回的梯度
     */
    public void upgradeWeight(float[][] w_deltas){
        for(int i = layers.size()-1; i >= 0; i--)
            layers.get(i).upgradeWeight(w_deltas[i]);
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

    private float loss(float[] x, float[] y_t){
        float[] y_pre = forward(x);
        loss = lossLayer.loss(y_pre, y_t);
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

        try {
            boolean newFile = file.createNewFile();
        } catch (Exception ignored) { }

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

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        if(!layers.isEmpty())
            layers.get(0).init(input_width, input_height, input_Dimension);
    }

    @Override
    public float[] forward(float[] inputs){
        ArrayList<float[]> output_list = forward_list(inputs);
        return output_list.get(output_list.size() - 1);
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
            if(back[1]==null) System.out.println(Arrays.toString(back[1]) + "   " + layers.get(i));
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
            int w_number = layers.get(i).getWeightNumber();
            if (w_number > 0) {
                float[] w_delta = new float[w_number];
                System.arraycopy(weightDeltas, index, w_delta, 0, w_number);
                index += w_number;
                layers.get(i).upgradeWeight(w_delta);
            }
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
    public int getWeightNumber_Train() {
        int n = 0;
        for (Layer layer : layers){
            n += layer.getWeightNumber_Train();
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

        pw.println(SaveData.sInt("layers_number", layers.size()));

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

        int layer_num = SaveData.getSInt(in.readLine());

        while ((line = in.readLine()) != null){
            Layer layer = getLayerById(SaveData.getSInt(line));
            layer.initByFile(in);
            layers.add(layer);

            if(layers.size() == layer_num)
                break;
        }

        setLearn_rate(learn_rate);
    }


    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();

        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        char[] c0 = new char[32 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "(" + this.output_width + ", " + this.output_height + ", " + this.output_dimension + ")";

        int v0 = 25 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');

        int param = this.getWeightNumber_Train();
        stringBuilder.append(name).append(c0).append(output_shape).append(c1).append(param);

        for (Layer layer: layers){
            stringBuilder.append("\n ").append(layer.toString());
        }

        return stringBuilder.toString();
    }


    public String summary() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append( "Sequential: ").append(EXPLAIN).append("\n")
                .append("_________________________________________________________________\n")
                .append(" Layer (type)               Output Shape Dimension       Param  \n")
                .append("=================================================================\n");


        int total_params_train = 0;
        int total_params = 0;
        for (Layer layer: layers){
           total_params += layer.getWeightNumber();
            total_params_train += layer.getWeightNumber_Train();
           stringBuilder.append(layer).append("\n");
        }


        stringBuilder.append("=================================================================\n")
                .append("Total train params: ").append(total_params_train).append("\n")
                .append("Total params: ").append(total_params).append("\n")
                .append("_________________________________________________________________");

        return  stringBuilder.toString();
    }

}
