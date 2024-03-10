package haili.deeplearn.function.loss;

import haili.deeplearn.function.Function;

//回归损伤函数
public class MSELoss extends Function
{

	public MSELoss(){
		super.id = 10;
	}

	@Override
	public float f(float y, float t)
	{
		// TODO: Implement this method
		float x = t - y;
		return x*x/0.5f;
	}

	@Override
	public float f_derivative(float y, float t)
	{
		// TODO: Implement this method
		return -(t - y);
	}


}
