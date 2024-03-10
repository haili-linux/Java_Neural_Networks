package haili.deeplearn.function.activation;

import haili.deeplearn.function.Function;
import haili.deeplearn.function.SourceCodeLib;

public class Tanh extends Function
{
	public Tanh(){
		super.id = 2;
		super.SourceCode = SourceCodeLib.tanh_code;
		super.name = SourceCodeLib.tanh_code_Name;
		super.SourceCode_derivative = SourceCodeLib.tanh_d_code;
		super.SourceCode_derivative_Name = SourceCodeLib.tanh_d_code_Name;
	}
	@Override
	public float f(float x)
	{
		// TODO: Implement this method
		float x1 = (float) Math.exp(x);
		float x2 = (float) Math.exp(-x);
		return (x1-x2)/(x1+x2);
	}

	@Override
	public float f_derivative(float x1)
	{
		// TODO: Implement this method
		//float x1 = f(x);
		return 1 - x1*x1;
	}


}
