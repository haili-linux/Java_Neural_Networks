package haili.deeplearn.utils;

public class MatrixUtil {


    public static float dot(float[] x1, float[] x2){
        if(x1.length != x2.length){
            System.out.println("MatrixUtil.dot(): Error x1.length != x2.length");
            try {
                throw new Exception("MatrixUtil.dot(): Error x1.length != x2.length");
            } catch (Exception exception) {
                exception.printStackTrace();
            }
            return 0;
        }

        float r = 0;
        for (int i = 0; i < x1.length; i++) r+= x1[i] * x2[i];

        return r;
    }


    public static float[] multi(float x1, float[] x2){
        float[] r = new float[x2.length];
        for (int i = 0; i < x2.length; i++)
            r[i] = x1 * x2[i];

        return r;
    }



    public static float[] add(float[] x1, float[] x2){
        if(x1.length != x2.length){
            System.out.println("MatrixUtil.add(): Error x1.length != x2.length");
            try {
                throw new Exception("MatrixUtil.add(): Error x1.length != x2.length");
            } catch (Exception exception) {
                exception.printStackTrace();
            }
            return null;
        }

        float[] r = new float[x1.length];
        for (int i = 0; i < r.length; i++)
            r[i] = x1[i] + x2[i];

        return r;
    }

    //数组合并
    public static float[] combine(float[] x1, float[] x2){
        float[] out = new float[x1.length + x2.length];

        System.arraycopy(x1, 0, out, 0, x1.length);
        System.arraycopy(x2, 0, out,  x1.length, x2.length);

        return out;
    }

    /**
     * 添加上位置编码信息
     * @param position 位置
     * @param vector 向量
     * @param value0 控制位置信息缩放，10000
     * @return
     */
    public static float[] addPositionEncoder(int position, float[] vector, int value0){
        for(int i = 0; i < vector.length; i++){

            double v0 = (double) (2 * i) / vector.length;
            double v1 = Math.pow(value0, v0);
            double v2 = position / v1;

            if(i % 2 == 0)
                vector[i] += (float) Math.sin(v2);
            else
                vector[i] += (float) Math.cos(v2);
        }
        return vector;
    }
    public static float[] removePositionEncoder(int position, float[] vector, int value0){
        for(int i = 0; i < vector.length; i++){

            double v0 = (double) (2 * i) / vector.length;
            double v1 = Math.pow(value0, v0);
            double v2 = position / v1;

            if(i % 2 == 0)
                vector[i] -= (float) Math.sin(v2);
            else
                vector[i] -= (float) Math.cos(v2);
        }
        return vector;
    }

    public static String ToString(float[] x, int h, int w){
        if( x == null || x.length == 0)
            return "[]";

        StringBuilder r = new StringBuilder("[" + x[0] + ", ");
        for (int i = 1; i < x.length - 1; i++){
            if((i+1)%w == 0)
                 r.append(x[i]).append("]\n[");
             else
                 r.append(x[i]).append(", ");
        }
        r.append(x[x.length - 1]).append("]");
        return r.toString();
    }

    public static String ArraysToString(float[] x, int w, int h){
        StringBuilder stringBuilder = new StringBuilder();
        for(int i = 1; i <= x.length; i++){
            stringBuilder.append(x[i-1]).append(", ");
            if(i % w == 0)
                stringBuilder.append("\n");

            if( i % (w * h) == 0)
                stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }

    public static float sum(float[] x){
        float r = 0;
        for(float xi : x)
            r += xi;
        return r;
    }
}
