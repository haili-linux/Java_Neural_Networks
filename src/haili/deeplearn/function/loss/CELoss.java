package haili.deeplearn.function.loss;

import haili.deeplearn.function.Function;

public class CELoss extends Function {
    //交叉熵
    public CELoss(){
        super.id = 11;
    }

    @Override
    public float f(float y, float t) {

        float v0 = 0;
        if(t != 0)
            v0 = (float) (t * Math.log10(y));

        float v1 = 0;
        if(t != 1)
            v1 = (float) ((1 - t) * Math.log10(1 - y));

        return -(v0 + v1);
    }

    @Override
    public float f_derivative(float y, float t) {

        float v0 = y - t;

        float r = 0;

        if(v0 != 0) {
            if (y != 1)
                r = v0 / (y * (1 - y));
            else
                r = 1000f;
        }

        return r;
    }

}
