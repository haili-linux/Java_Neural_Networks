package haili.deeplearn;

import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.Sigmoid;

import java.util.*;
import java.io.*;

public class Neuron implements Cloneable,Serializable
{
	// 单个神经元
	public int input_dimension = 1;//输入变量数


	public float[] w;//权值
	public float b = 0;//偏移量,阈值

	public Function ACT_function; // 激活函数

	//默认输入维度为1
	public Neuron(){
		w = new float[input_dimension];
		//b = 0;//偏置值随机
		init();
		ACT_function = new Sigmoid();
		//data_list = new float[input_n];
	}

	//指定输入维度
    public Neuron(int input_dimension){
		this.input_dimension = input_dimension;
		w = new float[input_dimension];
		//b = 0;
		init();
		ACT_function = new Sigmoid();
		//data_list = new float[input_n];
	}

	public Neuron(int input_dimension, Function activation){
		this.input_dimension = input_dimension;
		w = new float[input_dimension];
		//b = 0;
		init();
		ACT_function = activation;
		//data_list = new float[input_n];
	}

	//指定输入维度和偏置值
	public Neuron(int input_dimension, float b, Function activation){
		this.input_dimension = input_dimension;
		w = new float[input_dimension];
		this.b = b;
		init();
		ACT_function = activation;
		//data_list = new float[input_n];
	}

	//初始化
	private void init(){
		for(int i = 0; i < w.length; i++){
			//初始权值随机
			w[i] = (float) (Math.random() - 1f) * 2;
		}
	}
   
	public void setW(int n, float newW){
		w[n] = newW;
	}
	
	//激活函数
	public float act_f(float x){
		return ACT_function.f(x);
    }

	//严格定义in[],不然出错,结果记忆
	public float out(float[] in){
		float x = 0;
		for(int i = 0; i< input_dimension; i++)
			x += w[i] * in[i];
		x += b;
		//lastIn = x;
		//return  lastOut = ACT_function.f(x);
		return  ACT_function.f(x);
	}

	//不记忆结果
	public float out_notSave(float[] in){
		float x = 0;
		for(int i = 0; i< input_dimension; i++)
			x += w[i] * in[i];
        x += b;
		return  act_f(x);
	}

	public float LastIn_notSave(float[] in){
		float x = 0;
		for(int i = 0; i< input_dimension; i++)
			x += w[i] * in[i];
        x += b;
		return  x;
	}

	//扩展输入维度, n:要扩展的维度数,默认加在最后
	public void addInput_n(int add_number){
		if(add_number>0){
			input_dimension += add_number;
			float[] new_W = new float[input_dimension];
			float[] new_data_list = new float[input_dimension];
			
			for(int i=0;i<w.length;i++){
				new_W[i] = w[i];
				//new_data_list[i] = data_list[i];
			}
			float a = 0;
			for(int i = w.length; i< input_dimension; i++){
				new_W[i] = (float) Math.random();
				a += new_W[i];
		    }
			
			if(a>1)//防止过饱和
			  for(int i = w.length; i< input_dimension; i++)
			     new_W[i]/=a;
			
			w = new_W;
			//w_old = w;
			//data_list = new_data_list;
		}
	}

	//减少输入维度, n:要扩展的维度数,默认加在最后
	public void deleteInput_n(int de_number){
		if(de_number>0){
			input_dimension -= de_number;
			w = Arrays.copyOf(w, input_dimension);
		}
	}
	
	//设置输入维度
	public void newInput_n(int n){
		if(n>0){
		   input_dimension = n;
	       w = arraysOpon(w,new float[n]);
	    }
	}
	
	//数组映射
	public float[] arraysOpon(float[] a,float[] t){
		int al = a.length;
		int tl = t.length;
		if(al==tl) return a;
		float dindex;
		float index = 0;
		if(al>tl){
			dindex = (float)al/tl;
			for(int i=0;i<tl;i++){
				int j = floatToInt(index);
				if(j>=al)
					t[i] = a[al-1];
				else
				    t[i] = a[j];
				index += dindex;
			}
		}else{
			dindex = (float)tl/al;
			for(int i=0;i<al;i++){
				int j = floatToInt(index);
				if(j>=tl)
					t[tl-1] = a[i];
				else
					t[j] = a[i];
				index += dindex;
			}
		}
		return t;
	}

	//浮点数转整形，四舍五入
	final private int floatToInt(float x){
		float a = x%1;
		int r = (int)x;
		if(a>=0.5) r++;
		return r;
	}
	
	@Override
	public String toString() {
		// TODO: Implement this method
		return "w:" + Arrays.toString(w) + "  d:"+ b;
	}

	@Override
    protected Object clone()  {
        Neuron p =null;
        try {
            p= (Neuron)super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return p;
    }
	public Object deepClone() throws IOException, ClassNotFoundException{
		ByteArrayOutputStream bo = new ByteArrayOutputStream();
		ObjectOutputStream os = new ObjectOutputStream(bo);
		os.writeObject(this);

		ByteArrayInputStream bi = new ByteArrayInputStream(bo.toByteArray());
		ObjectInputStream is = new ObjectInputStream(bi);
		return is.readObject();
	}
}
