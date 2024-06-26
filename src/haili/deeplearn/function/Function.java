package haili.deeplearn.function;


import haili.deeplearn.function.activation.*;
import haili.deeplearn.function.loss.CELoss;
import haili.deeplearn.function.loss.MSELoss;

public class Function
{

	public int id = 0;
	public String name;
	public String SourceCode;
	public String SourceCode_derivative;
	public String SourceCode_derivative_Name;
    public float f(float x){ return x; }
	public float f(float x1,float x2){return 0;}

	public float f_derivative(float out){ return 1; }

	public float f_derivative(float x1, float x2){ return 0; }

	public static Function getFunctionById(int id){
		Function r;
		switch (id){
			case 1: r = new Sigmoid(); break;
			case 2: r = new Tanh();    break;
			case 3: r = new Relu();    break;
			case 4: r = new LRelu();   break;
			case 5: r = new Softmax(); break;

			case 10:r = new MSELoss(); break;
			case 11:r = new CELoss();  break;

			default:r = new Function(); break;
		}
		return r;
	}

	@Override
	public String toString() {
		return super.toString() + "BaseFunction ";
	}

}
