package haili.deeplearn.function;

public class Sigmoid extends Fuction
{
	public Sigmoid(){
		super.id = 1;
		super.SourceCode = SourceCodeLib.sigmoid_code;
		super.name = SourceCodeLib.sigmoid_code_Name;
		super.SourceCode_derivative = SourceCodeLib.sigmoid_d_code;
		super.SourceCode_derivative_Name = SourceCodeLib.sigmoid_d_code_Name;
	}
	
	@Override
	public float f(float x)
	{
		// TODO: Implement this method
		return (float)(1.0f / (1.0f + Math.exp(-x)));
	}

	@Override
	public float f_derivative(float x1)
	{
		// TODO: Implement this method
		//float x1 = f(x);
		return x1 * (1.0f - x1);
	}
}
