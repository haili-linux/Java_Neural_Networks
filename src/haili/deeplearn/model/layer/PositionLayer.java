package haili.deeplearn.model.layer;

import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.utils.SaveData;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

public class PositionLayer extends Layer{

    public float[][] positionCode;

    public PositionLayer(int one_inputVector_dimension ,int positionCodeVector, int maxPosition){
        this.id = 13;

        positionCode = new float[maxPosition][positionCodeVector];
        for(int i = 0; i < maxPosition; i++)
            Arrays.fill(positionCode[i],  (i + 1f) / maxPosition );

        this.input_width = one_inputVector_dimension;
        this.input_height = 1;
        this.input_dimension = input_width;

        this.output_width = this.input_width + positionCode[0].length;
        this.output_height = 1;
        this.output_dimension = this.output_width;
    }

    @Override
    public void init(int input_width, int input_height, int input_Dimension) {

    }


    /**
     *
     * @param inputs :
     * @return {seq0:{i0, i1, i2, i3,..., iN, position0,...,positionN}, seq1:{i0, i1, i2, i3,..., iN, position0,..., positionN}, ..., seqN }
     */
    @Override
    public float[] forward(float[] inputs) {
        int seqLen = inputs.length / input_width;

        float[] outputs = new float[output_width * seqLen];

        for (int seqi = 0; seqi < seqLen; seqi++){
            System.arraycopy(inputs, seqi * input_width, outputs, seqi * output_width, input_width);
            //for(int i = 0; i < input_width; i++)
            //    outputs[seqi * output_width + i] = inputs[seqi * input_width + i];

            //int positionCodeLen = positionCode[0].length;
            float[] pi = positionCode[seqi];

            System.arraycopy(pi, 0, outputs, seqi * output_width + input_width, pi.length);
            //for(int i = 0; i < pi.length; i++)
            //    outputs[seqi * output_width + input_width + i] = pi[i];
        }
        return outputs;
    }


    @Override
    public float[][] backward(float[] inputs, float[] output, float[] deltas) {
        float[] w_deltas = new float[getWeightNumber()];
        float[] inputs_deltas = new float[inputs.length];

        int pLen = positionCode[0].length;

        int seqLen = inputs.length / input_width;

        for(int seqi = 0; seqi < seqLen; seqi++){
            System.arraycopy(deltas, seqi * output_width, inputs_deltas, seqi * input_width, input_width);
            //for(int i = 0; i < input_width; i++)
            //    inputs_deltas[seqi * input_width + i] = deltas[seqi * output_width + i];

            System.arraycopy(deltas, seqi * output_width + input_width, w_deltas, seqi * pLen, pLen);
            //for(int i = 0; i < pLen; i++)
            //    w_deltas[seqi * pLen + i] = deltas[seqi * output_width + input_width + i];
        }

        return new float[][]{inputs_deltas, w_deltas};
    }


    @Override
    public int getWeightNumber() {
        return positionCode.length * positionCode[0].length;
    }


    @Override
    public int getWeightNumber_Train() {
        return getWeightNumber();
    }


    @Override
    public void setDeltaOptimizer(BaseOptimizerInterface deltaOptimizer) {
        deltaOptimizer = deltaOptimizer.getNewObject();
        deltaOptimizer.init(getWeightNumber());
        super.setDeltaOptimizer(deltaOptimizer);
    }


    @Override
    public void upgradeWeight(float[] weightDeltas) {
        int index = 0;
       for(int i = 0; i < positionCode.length; i++)
           for (int j = 0; j < positionCode[i].length; j++){
               positionCode[i][j] -= learn_rate * deltaOptimizer.DELTA(weightDeltas[index], index);
               index++;
           }
    }

    @Override
    public void saveInFile(PrintWriter pw) throws Exception {
        pw.println(SaveData.sInt("Layer_ID", id));

        pw.println(SaveData.sInt("one_inputVector_dimension", input_width));
        pw.println(SaveData.sInt("maxPosition", positionCode.length));

        for(int i = 0; i < positionCode.length; i++)
            pw.println(SaveData.sFloatArrays("p["+ i +"]",positionCode[i]));
    }

    @Override
    public void initByFile(BufferedReader in) throws Exception {
        int one_inputVector_dimension = SaveData.getSInt(in.readLine());
        int maxPosition = SaveData.getSInt(in.readLine());
        positionCode = new float[maxPosition][];

        for(int i = 0 ; i < maxPosition; i++)
            positionCode[i] = SaveData.getsFloatArrays(in.readLine());

        this.input_width = one_inputVector_dimension;
        this.input_height = 1;
        this.input_dimension = input_width;

        this.output_width = this.input_width + positionCode[0].length;
        this.output_height = 1;
        this.output_dimension = this.output_width;
    }


    @Override
    public String toString() {
        StringBuilder stringBuilder= new StringBuilder();
        String name = this.getClass().getName();
        name = " " + name.substring(name.lastIndexOf(".") + 1);

        char[] c0 = new char[27 - name.length()];
        Arrays.fill(c0, ' ');

        String output_shape = "in:(N, " + input_dimension + ")  out:(N, " + output_dimension + ")  ";

        int v0 = 30 - output_shape.length();
        if(v0 < 1) v0 = 1;
        char[] c1 = new char[v0];
        Arrays.fill(c1, ' ');

        int param = getWeightNumber_Train();

        char[] c2 = new char[0];
        if(deepOfSequential > 0) {
            c2 = new char[deepOfSequential * 2];
            Arrays.fill(c2, ' ');
        }

        stringBuilder.append(c2).append(name).append(c0).append(output_shape).append(c1).append(param);

        return stringBuilder.toString();
    }
}
