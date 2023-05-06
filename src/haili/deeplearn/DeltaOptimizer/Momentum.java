package haili.deeplearn.DeltaOptimizer;

public class Momentum implements BaseOptimizerInterface {
    float nl, dnl;
    float[] m;

    public Momentum(int delta_number , float nl) {
        m = new float[delta_number];
        this.nl = nl;
        dnl = 1.0f - nl;
    }

    @Override
    public void init() { }

    @Override
    public float DELTA(float delta, int index) {
         m[index] =  nl * m[index] + dnl * delta;
         return  m[index];
    }
}
