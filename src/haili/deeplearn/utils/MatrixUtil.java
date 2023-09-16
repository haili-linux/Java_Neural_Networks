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
        for (int i = 0; i < x2.length; i++) x2[i] *= x1;

        return x2;
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

        for (int i = 0; i < x1.length; i++) x1[i] += x2[i];

        return x1;
    }

    public static float[] combine(float[] x1, float[] x2){
        float[] out = new float[x1.length + x2.length];

        System.arraycopy(x1, 0, out, 0, x1.length);
        System.arraycopy(x1, 0, out, 0 + x1.length, x2.length);

        return out;
    }

}
