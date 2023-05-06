package haili.deeplearn.DeltaOptimizer;

public class BaseOptimizer implements BaseOptimizerInterface {
    @Override
    public void init() {
    }
    @Override
    public float DELTA(float delta, int index) {
        return delta;
    }
}
