package haili.deeplearn.function.loss;

import haili.deeplearn.function.Function;

public class CESLoss extends Function {
    public CESLoss(){
        super.id = 12;
    }

    @Override
    public float f(float y, float t) {
        return -t * (float) Math.log10(y);
    }

    @Override
    public float f_derivative(float y, float t) {
           return -t/y;
    }
}
