package haili.deeplearn.function;

public class LRelu extends Fuction {
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
