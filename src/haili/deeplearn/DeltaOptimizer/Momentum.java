package haili.deeplearn.DeltaOptimizer;

public class Momentum implements BaseOptimizerInterface {
    float nl = -1, dnl = -1;
    float[] m;

    public Momentum(int delta_number , float nl) {
        this.nl = nl;
        dnl = 1.0f - nl;

        init(delta_number);
    }

    public Momentum(float nl){
        this.nl = nl;
        dnl = 1.0f - nl;
    }

    public Momentum(){
        nl = 0.9f;
        dnl = 0.1f;
    }

    @Override
    public void init(int wn) { m = new float[wn]; }

    @Override
    public float DELTA(float delta, int index) {
         m[index] =  nl * m[index] + dnl * delta;
         return  m[index];
    }

    @Override
    public BaseOptimizerInterface getNewObject() {
        if( nl != -1 && dnl != -1)
            return new Momentum(nl);
        else
            return new Momentum();
    }
}
