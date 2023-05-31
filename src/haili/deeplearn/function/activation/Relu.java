package haili.deeplearn.function.activation;

import haili.deeplearn.function.Fuction;
import haili.deeplearn.function.SourceCodeLib;

public class Relu extends Fuction
{
	public Relu(){
		super.id = 3;
		super.SourceCode = SourceCodeLib.relu_code;
		super.name = SourceCodeLib.relu_code_Name;
		super.SourceCode_derivative = SourceCodeLib.rule_d_code;
		super.SourceCode_derivative_Name = SourceCodeLib.rule_d_code_Name;
	}
	@Override
	public float f(float x)
	{
		// TODO: Implement this method
		return (x>0) ? x:0;
	}

	@Override
	public float f_derivative(float x)
	{
		// TODO: Implement this method
		return (x>0) ? 1:0;
	}


}

