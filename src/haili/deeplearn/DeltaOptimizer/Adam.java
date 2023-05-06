package haili.deeplearn.DeltaOptimizer;

public class Adam implements BaseOptimizerInterface {

    final float v1;
    final float v2;
    final float e;
    float dv1;
    float dv2;
    float v1_t;
    float v2_t;
    float[] m;
    float[] v;
    int len;
    private int p;

    public Adam(int delta_number, float v1, float v2, float e){
        m = new float[delta_number];
        v = new float[delta_number];
        len = m.length -1;
        this.v1 = v1;
        this.v2 = v2;
        this.e = e;
        dv1 = 1 - v1;
        dv2 = 1 - v2;
        v1_t = v1;
        v2_t = v2;
        p = 0;
    }

    @Override
    public void init() {
        v1_t = v1;
        v2_t = v2;
        p = 0;
        m = new float[m.length];
        v = new float[v.length];
    }

    @Override
    public synchronized float DELTA(float delta, int index) {

        if(p == len) {
            p = 0;
            v1_t *= v1;
            v2_t *= v2;
        } else {
            p++;
        }

        m[index] = v1 * m[index] + dv1 * delta;
        v[index] = v2 * v[index] + dv2 * delta * delta;

        float m_ = m[index] / (1 - v1_t);
        float v_ = v[index] / (1 - v2_t);

        return (float)(m_ / ( Math.sqrt(v_) + e ));
    }

}
