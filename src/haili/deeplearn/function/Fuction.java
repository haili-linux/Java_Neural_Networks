package haili.deeplearn.function;


public class Fuction
{

	public int id = 0;
	public String name;
	public String SourceCode;
	public String SourceCode_derivative;
	public String SourceCode_derivative_Name;
    public float f(float x){ return 0; }
	public float f(float x1,float x2){return 0;}
	public float f_derivative(float x){ return 0; }
	public float f_derivative(float x1, float x2){ return 0; }

	public static Fuction getFunctionById(int id){
		Fuction r;
		switch (id){
			case 1: r = new Sigmoid(); break;
			case 2: r = new Tanh();    break;
			case 3: r = new Relu();    break;
			case 4: r = new LRelu();   break;

			case 10:r = new MSELoss(); break;
			case 11:r = new CELoss();  break;
			case 12:r = new CESLoss(); break;
			default:r = new Fuction(); break;
		}
		return r;
	}

}
