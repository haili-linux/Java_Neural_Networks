package haili.deeplearn.function;

public class CELoss extends Fuction {
    //交叉熵
    public CELoss(){
        super.id = 11;
    }

    @Override
    public float f(float y, float t) {
        return (float)-(t*Math.log10(y) + (1-t)*Math.log10(1-y));
    }

    @Override
    public float f_derivative(float y, float t) {
        return (y-t)/(y*(1-y));
    }

}
