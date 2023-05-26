package haili.deeplearn.DeltaOptimizer;

public class BaseOptimizer implements BaseOptimizerInterface {
    @Override
    public void init(int wn) {
    }
    @Override
    public float DELTA(float delta, int index) {
        return delta;
    }

    @Override
    public BaseOptimizerInterface getNewObject() {
        return new BaseOptimizer();
    }
}
