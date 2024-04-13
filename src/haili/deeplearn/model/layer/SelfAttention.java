package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.function.Function;
import haili.deeplearn.model.layer.softmax.SoftmaxLayer;
import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;

import static java.lang.Float.NaN;


public class SelfAttention extends Layer{


    public Dense q_layer, k_layer, v_layer;
    public Layer scoreLayer = new SoftmaxLayer();

    /**
     * @param one_input_vector_dimension 一个单位输入向量的维度
     * @param one_output_vector_dimension 一个单位输出向量的维度
     * @param Q_K_dimension q k 的大小
     */
    public SelfAttention(int one_input_vector_dimension, int one_output_vector_dimension, int Q_K_dimension){
        this.id = 7;
        this.input_width = one_input_vector_dimension;
        this.output_width = one_output_vector_dimension;
        this.output_dimension = one_output_vector_dimension;

        q_layer = new Dense(one_input_vector_dimension,1, Q_K_dimension,1, Q_K_dimension, new Function(), false);
        k_layer = new Dense(one_input_vector_dimension, 1, Q_K_dimension, 1, Q_K_dimension, new Function(), false);
        v_layer = new Dense(one_input_vector_dimension, 1, one_output_vector_dimension, 1, one_output_vector_dimension, new Function(), false);

        init(0, 0, 0);
    }


    @Override
    public void init(int input_width, int input_height, int input_Dimension) {
        this.input_dimension = this.input_width;
        this.output_dimension = this.output_width;
    }

    public float[][][] forward_list(float[] inputs){
        float[][][] output = null;
        if(saveHiddenLayerOutput && hiddenLayerOutputMap.containsKey(inputs)) {
            output = (float[][][]) hiddenLayerOutputMap.get(inputs);
            if(output != null)
                return output;
        }

        //计算
        output = forwardList(inputs);

        // 保存中间输出
        if(saveHiddenLayerOutput){
            hiddenLayerOutputMap.put(inputs, output);
        }
        return output;
    }

    private float[][][] forwardList(float[] inputs){
        float[][][] r = new float[6][][];
        float sqrt_d =  (float) Math.sqrt(q_layer.output_dimension);

        // 当前输入的seq数量
        int seqLen = inputs.length / input_width;
        float[] outputs = new float[seqLen * output_width];
        float[][] inputs_ = new float[seqLen][];

        // 计算QKV
        float[][] q = new float[seqLen][];
        float[][] k = new float[seqLen][];
        float[][] v = new float[seqLen][];
        for(int i = 0; i < seqLen; i++){
            inputs_[i] =  new float[input_width];
            System.arraycopy(inputs, i * input_width, inputs_[i], 0, input_width);

            q[i] = q_layer.forward(inputs_[i]);
            k[i] = k_layer.forward(inputs_[i]);
            v[i] = v_layer.forward(inputs_[i]);
        }


        // 计算 Q * K
        float[][] score = new float[seqLen][seqLen];
        for(int i = 0; i < seqLen; i++) {
            for(int j = 0; j < seqLen; j++) {
                score[i][j] = MatrixUtil.dot(q[i], k[j]) / sqrt_d;
            }

            // 对相关分数处理，默认经过softmax
            score[i] = scoreLayer.forward(score[i]);

            // 计算输出
            int od =  i * output_width;
            for(int ik = 0; ik < output_width; ik++) {
                for (int ij = 0; ij < seqLen; ij++) {
                    outputs[od + ik] +=  (score[i][ij] * v[ij][ik]);
                }
            }
        }


        r[0] = inputs_;
        r[1] = q;
        r[2] = k;
        r[3] = v;
        r[4] = score;
        r[5] = new float[][]{outputs};
        return r;
    }


    @Override
    public float[] forward(float[] inputs) {
        return forward_list(inputs)[5][0];
    }

    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        // 当前输入的seq数量
        int seqLen = deltas.length / output_width;

        float sqrt_d =  (float) Math.sqrt(q_layer.output_dimension);

        float[][][] forwardOut = forward_list(inputs);
        float[][] inputs_ = forwardOut[0];
        float[][] q = forwardOut[1];
        float[][] k = forwardOut[2];
        float[][] v = forwardOut[3];
        float[][] score = forwardOut[4];

        float[][] q_deltas = new float[seqLen][q_layer.output_dimension];
        float[][] k_deltas = new float[seqLen][k_layer.output_dimension];
        float[][] v_deltas = new float[seqLen][output_width];

        float[][] score_deltas = new float[seqLen][seqLen];
        for(int i = 0; i < seqLen; i++) {
            // 计算输出
            int od =  i * output_width;
            for(int ik = 0; ik < output_width; ik++) {
                for (int ij = 0; ij < seqLen; ij++) {
                    score_deltas[i][ij] +=  v[ij][ik] * deltas[od + ik] / sqrt_d;
                    v_deltas[ij][ik] += score[i][ij] * deltas[od + ik] / sqrt_d;
                }
            }

            score_deltas[i] = scoreLayer.backward(null, score[i],  score_deltas[i])[0];

            for(int j = 0; j < seqLen; j++) {
                for(int ij = 0; ij < q_layer.output_dimension; ij++) {
                    q_deltas[i][ij] += k[j][ij] * score_deltas[i][j];
                    k_deltas[j][ij] += q[i][ij] * score_deltas[i][j];
                }
            }
        }


        float[] last_layer_deltas = new float[inputs.length];

        //{wq, wk, wv}
        float[] w_deltas = new float[q_layer.getWeightNumber() + k_layer.getWeightNumber() + v_layer.getWeightNumber()];

        int dk = q_layer.getWeightNumber();
        int dv = q_layer.getWeightNumber() + k_layer.getWeightNumber();
        for(int i = 0; i < seqLen; i++) {
            float[][] backs_q = q_layer.backward(inputs_[i], q[i], q_deltas[i]);
            float[][] backs_k = k_layer.backward(inputs_[i], k[i], k_deltas[i]);
            float[][] backs_v = v_layer.backward(inputs_[i], v[i], v_deltas[i]);

            for(int j = 0; j < q_layer.getWeightNumber(); j++){
                w_deltas[j] += backs_q[1][j];
                w_deltas[dk + j] += backs_k[1][j];
            }

            for(int j = 0; j < v_layer.getWeightNumber(); j++){
                w_deltas[dv + j] += backs_v[1][j];
            }

            int d = i * input_width;
            for(int j = 0; j < input_width; j++){
                last_layer_deltas[d + j] = backs_q[0][j] + backs_k[0][j] + backs_v[0][j];
            }
        }

        return new float[][]{last_layer_deltas, w_deltas};
    }

    @Override
    public void upgradeWeight(float[] weightDeltas) {
        float[] wq = new float[q_layer.getWeightNumber()];
        float[] wk = new float[k_layer.getWeightNumber()];
        float[] wv = new float[v_layer.getWeightNumber()];

        System.arraycopy(weightDeltas, 0, wq, 0, wq.length);

        System.arraycopy(weightDeltas, wq.length, wk, 0, wk.length);

        int dv = wq.length + wk.length;
        System.arraycopy(weightDeltas, dv, wv, 0, wv.length);

        q_layer.upgradeWeight(wq);
        k_layer.upgradeWeight(wk);
        v_layer.upgradeWeight(wv);
    }

    int wn = -1;
    @Override
    public int getWeightNumber() {
        if(wn == -1)
            wn = q_layer.getWeightNumber() + k_layer.getWeightNumber() + v_layer.getWeightNumber();
        return  wn;
    }


    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        deltaOptimizer = deltaOptimizer.getNewObject();
        deltaOptimizer.init(getWeightNumber() );
        super.setDeltaOptimizer(deltaOptimizer);
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("input_width", input_width));
        pw.println(SaveData.sInt("output_width", output_width));

        q_layer.saveInFile(pw);
        k_layer.saveInFile(pw);
        v_layer.saveInFile(pw);
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        input_width = SaveData.getSInt(in.readLine());
        output_width = SaveData.getSInt(in.readLine());

        q_layer = (Dense) getLayerById(SaveData.getSInt(in.readLine()));
        q_layer.initByFile(in);

        k_layer = (Dense) getLayerById(SaveData.getSInt(in.readLine()));
        k_layer.initByFile(in);

        v_layer = (Dense) getLayerById(SaveData.getSInt(in.readLine()));
        v_layer.initByFile(in);
    }
}
