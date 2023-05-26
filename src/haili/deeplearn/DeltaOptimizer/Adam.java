package haili.deeplearn.DeltaOptimizer;

public class Adam implements BaseOptimizerInterface {

    float v1 = -1;
    float v2 = -1;
    float e = -1;
    float dv1;
    float dv2;
    float v1_t;
    float v2_t;
    float[] m;
    float[] v;
    int len;
    //private int p;

    public Adam(int delta_number, float v1, float v2, float e){

        this.v1 = v1;
        this.v2 = v2;
        this.e = e;

       init(delta_number);
        //p = 0;
    }


    public Adam(float v1, float v2, float e){
        this.v1 = v1;
        this.v2 = v2;
        this.e = e;
    }


    public Adam(){
        v1 = 0.9f;
        v2 = 0.999f;
        e = 1e-8f;
    }



    @Override
    public void init(int wn) {
        m = new float[wn];
        v = new float[wn];
        len = m.length -1;

        v1_t = v1;
        v2_t = v2;
        dv1 = 1 - v1;
        dv2 = 1 - v2;
    }

    @Override
    public synchronized float DELTA(float delta, int index) {
        /*
        if(p == len) {
            p = 0;
            v1_t *= v1;
            v2_t *= v2;
        } else {
            p++;
        }*/

        m[index] = v1 * m[index] + dv1 * delta;
        v[index] = v2 * v[index] + dv2 * delta * delta;

        float m_ = m[index] / (1 - v1_t);
        float v_ = v[index] / (1 - v2_t);

        return (float)(m_ / ( Math.sqrt(v_) + e ));
    }


    @Override
    public BaseOptimizerInterface getNewObject() {
        if(v1 != -1 && v2 != -1 && e != -1 )
            return new Adam(v1, v2, e);
         else
            return new Adam();
    }
}
