package haili.deeplearn.function.activation;

import haili.deeplearn.function.Function;
import haili.deeplearn.function.SourceCodeLib;

public class LRelu extends Function {
    public  LRelu(){
        super.id = 4;
        super.SourceCode = SourceCodeLib.Lrelu_code;
        super.name = SourceCodeLib.Lrelu_code_Name;
        super.SourceCode_derivative = SourceCodeLib.lrule_d_code;
        super.SourceCode_derivative_Name = SourceCodeLib.lrule_d_code_Name;
    }
    @Override
    public float f(float x) {
        return (x>0) ? x : 0.001f*x;
    }

    @Override
    public float f_derivative(float x) {
        return (x>0) ? 1 : 0.001f;
    }

}
