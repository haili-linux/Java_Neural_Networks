package haili.deeplearn;

import haili.deeplearn.function.*;
import haili.deeplearn.function.loss.MSELoss;
import haili.deeplearn.utils.DataSetUtils;
import haili.deeplearn.utils.SaveData;
import haili.deeplearn.utils.ThreadWork;
import haili.deeplearn.DeltaOptimizer.BaseOptimizer;
import haili.deeplearn.DeltaOptimizer.BaseOptimizerInterface;
import haili.deeplearn.utils.ProgressBarCmd;

import java.io.*;
import java.util.*;

public class BpNetwork extends SaveData implements Cloneable,Serializable
{
	public int input_n;//输入维度
	public int output_n;//输出维度
	public float lr;//学习率 0～1
	public int act_fuctiom_ID;//网络的激活函数
	public int[] hid_n; //隐藏层神经元结构

	public float dError; //误差
	public Fuction Loss_function;

	//public float[] inNeuer_out;//输入层每个神经元的输出
	public float[] outNeuer_out;//输出
	public float[][] hiddenNeuer_out;//隐藏层每个输出

	//public Neuer[] input_Neuer;//输入层神经元
	public Neuron[] output_Neuron;//输出层神精元
	public Neuron[][] hidden_Neuron;/*第i层*//*i层神经元数量*///隐藏层神经元

	public String EXPLAIN; //神经网络说明

	//ExecutorService upThreadPool;//用于并行计算的线程池

	public BaseOptimizerInterface deltaOptimizer; //梯度优化器

	//in:输入量维度, outn:输出结果维度, ng:学习效率  hidden_:隐藏层每层神经元数量
	public BpNetwork(int in_vector, int out_vector, float ng, Fuction act_fuction , int[] hidden_){
		input_n = in_vector;
		output_n = out_vector;
		lr = ng;
		hid_n = hidden_;
		act_fuctiom_ID = act_fuction.id;

		init(act_fuction);
		//upThreadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		deltaOptimizer = new BaseOptimizer();
	}

	//从文件初始化
	public BpNetwork(String file){
		readFile(file);
		System.out.println("explain:"+EXPLAIN);
		System.out.println("input dimension: " + input_n);
		System.out.println("output dimension: " + output_n);
		System.out.println("loss: " + dError);
		Fuction[] fuctions = new Fuction[hid_n.length + 1];
		for (int i = 0; i < hid_n.length; i++){
			fuctions[i] =  hidden_Neuron[i][0].ACT_function;
		}
		fuctions[hid_n.length] = output_Neuron[0].ACT_function;
		System.out.println("activation function: " + Arrays.toString(fuctions));
		System.out.println("hidden layer: " + Arrays.toString(hid_n));
		deltaOptimizer = new BaseOptimizer();
		//upThreadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
	}
	
	//计算输出
	public float[] out_(float[] in){
		if(in.length!=input_n){
		    System.out.println("输入数据格式错误");
			return null;
		}

		if(_in_max>1)//归一化
		   for(int i=0;i<in.length;i++)
		       in[i] /= _in_max;
			   
		float[] r = out_no_Save(in);
		
		if(_out_max>1)
		  for(int i=0;i<r.length;i++)
		    r[i] *= _out_max;
		
		return r;
	}


	//训练模型
	public float fit(float[][] train_X,float[][] train_Y, int batch_size, int epoch, int Thread_n){
		//参数检查
		if(batch_size<1 &&epoch <0) return 99999;
		//获取cpu核心数
		int core_number = Runtime.getRuntime().availableProcessors();
		if(Thread_n>core_number) Thread_n = core_number;

		/*
		if(batch_size==1) {//batch_size=1,随机梯度下降
			for (int i = 0; i < epoch; i++) {
				String title = "  epoch: " + i + "  ";
				ProgressBarCmd progressBarCmd = new ProgressBarCmd(title, train_X.length, 50);
				for (int j = 0; j < train_X.length; j++) {
					upgrade(train_X[j], train_Y[j]);
					System.out.print(progressBarCmd.setProgress(i + 1));
				}
			}
		} else*/
		if(batch_size >= train_X.length) {//batch_size和训练集一样，全批量梯度下降
			for (int i = 0; i < epoch; i++) {
				upgrade(train_X, train_Y, Thread_n);
			}
		} else {//mini-batch

			ArrayList<float[][]>[] data = DataSetUtils.splitBatch(train_X, train_Y, batch_size);
			ArrayList<float[][]> train_x = data[0];
			ArrayList<float[][]> train_y = data[1];

			for (int i = 0; i < epoch; i++){
				String title = "  epoch: " + (i + 1) + "  ";

				upgrade_mini_batch_progressbar(Thread_n, train_x, train_y, title);
			}
		}
		//计算误差
		return dError = calculateLoss(train_X, train_Y);
	}


	public float fit_time(float[][] train_X,float[][] train_Y,int batch_size,int Thread_n,int time_min){
		//参数检查
		if(batch_size<1 && time_min <0) return 99999;
		//获取cpu核心数
		int core_number = Runtime.getRuntime().availableProcessors();
		if(Thread_n>core_number) Thread_n = core_number;

		long startTime = System.currentTimeMillis();

		/*if(batch_size==1)//batch_size=1,随机梯度下降
		{
			while (true) {
				for (int j = 0; j < train_X.length; j++) upgrade(train_X[j], train_Y[j]);
				long Time = (System.currentTimeMillis() - startTime) / 60000;
				if(Time >= time_min) break;
			}
		}else */

		if(batch_size >= train_X.length)//batch_size和训练集一样，全批量梯度下降
		{
			int epoch = 0;
			System.out.print("  epoch: " + epoch);
			while (true) {
				epoch++;

				char[] tb = new char[(epoch+"").length()];
				Arrays.fill(tb, '\b');
				System.out.print(new String(tb) + epoch);

				upgrade(train_X, train_Y, Thread_n);

				long Time = (System.currentTimeMillis() - startTime) / 60000;
				if(Time >= time_min) break;
			}
		}
		else//mini-batch
		{
			ArrayList<float[][]>[] data = DataSetUtils.splitBatch(train_X, train_Y, batch_size);
			ArrayList<float[][]> train_x = data[0];
			ArrayList<float[][]> train_y = data[1];

			long Time ;
			int epoch = 0;
			//System.out.print("  epoch: " + epoch);
			while (true) {

				epoch++;

				String title = "  epoch: " + epoch + "  ";

				upgrade_mini_batch_progressbar(Thread_n, train_x, train_y, title);

				Time = (System.currentTimeMillis() - startTime) / 60000;
				if (Time >= time_min) break;
			}

		}
		//计算误差
		return dError = calculateLoss(train_X, train_Y);
	}

	public float fit(float[][] train_X,float[][] train_Y, int epoch, int batch_size){
		return fit(train_X, train_Y, batch_size, epoch,  Runtime.getRuntime().availableProcessors());
	}

	private void upgrade_mini_batch_progressbar(int Thread_n, ArrayList<float[][]> train_x, ArrayList<float[][]> train_y, String title) {
		ProgressBarCmd progressBarCmd = new ProgressBarCmd(title, train_x.size(), 50);
		System.out.print(progressBarCmd.setProgress(0));
		for (int i = 0; i < train_x.size(); i++) {
			upgrade(train_x.get(i), train_y.get(i), Thread_n);
			System.out.print(progressBarCmd.setProgress(i + 1));
		}
	}

	//测试一个数据集上的误差
	public float calculateLoss(float[][] train_X, float[][] train_Y){

		ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(train_X.length){

			final float[] loss = new float[train_X.length];

			@Override
			public void working(int index) {
				loss[index] = Loss(out_no_Save(train_X[index]), train_Y[index]);
			}

			@Override
			public Object getObject() {
				return loss;
			}
		};

		ThreadWork.start(threadWorker);

		float error = 0;
		for (float loss : (float[]) threadWorker.getObject()){
			error += loss;
		}

		return error / train_X.length;

	}
	
	//数据归一化
	float _in_max=1.0f,_out_max=1.0f;//选出训练集中最大值
	public float[][][] data_of_one(float[][] input,float[][] output){
		_in_max = getListMax(input);
		_out_max = getListMax(output);
	    if(_in_max>1)
			for(int i=0;i<input.length;i++)
				for(int j=0;j<input[i].length;j++)
					input[i][j] = input[i][j]/_in_max;
		else _in_max = 1;
		if(_out_max>1)
			for(int i=0;i<output.length;i++)
				for(int j=0;j<output[i].length;j++)
					output[i][j] = output[i][j]/_out_max;
		else _out_max = 1;
		return new float[][][]{input,output};
	}

	 
	//增加输入维度, n:要扩展的维度数,默认加在最后
	/*public void addInput_n(int add_number,Fuction act_fuction){
		if(add_number>0){
		   input_n += add_number;
		   ArrayList<Neuer> inputNeuer =new ArrayList<Neuer>(Arrays.asList(input_Neuer));
		   input_Neuer = new Neuer[input_n];
		   inNeuer_out = new float[input_n];
		   int i;
		   //复制
		   for(i=0;i<inputNeuer.size();i++)
			   input_Neuer[i] = inputNeuer.get(i);
		   //新增
		   for(i=inputNeuer.size();i<input_n;i++)
		       input_Neuer[i] = new Neuer(1,act_fuction);
			 
		   //调整下一层神经元输入数
		   for(i=0;i<hidden_Neuer[0].length;i++)
			   hidden_Neuer[0][i].addInput_n(add_number);
		   
		}
	}

	public void addInput_n(int add_number){
		addInput_n(add_number,getFuctionById(act_fuctiom_ID));
	}

	//减少输入维度, n:要减的维度数,默认加在最后
	public void deleteInput_n(int de_number){
		if(de_number>0 && input_n>de_number){
		     input_n -= de_number;
		     input_Neuer = Arrays.copyOf(input_Neuer,input_n);
		     inNeuer_out = new float[input_n];
		  
		     //下一层
		     for(int i=0;i<hidden_Neuer[0].length;i++)
			     hidden_Neuer[0][i].deleteInput_n(de_number);
		}
	}*/
	
	//扩展输出维度,默认在最后
	public void addOutput_n(int add_number, Fuction act_fuction){
		if(add_number>0){
			output_n += add_number;
			Neuron[] newOutput_Neuron = new Neuron[output_n];
			
			int i;
			for(i=0; i< output_Neuron.length; i++)
			     newOutput_Neuron[i] = output_Neuron[i];
			
			for(i= output_Neuron.length; i<output_n; i++)
			     newOutput_Neuron[i] = new Neuron(hidden_Neuron[hidden_Neuron.length-1].length,act_fuction);
			
			output_Neuron = newOutput_Neuron;
			outNeuer_out = new float[output_n];	
		}
	}
	public void addOutput_n(int add_number){
		addOutput_n(add_number, Fuction.getFunctionById(act_fuctiom_ID));
	}

	//减少输出维度, n:要减的维度数,默认加在最后
	public void deleteOutput_n(int de_number){
		if(de_number>0 && output_n>de_number){
			output_n -= de_number;
			outNeuer_out = new float[output_n];
			output_Neuron = Arrays.copyOf(output_Neuron,output_n);
		}
	}


	//在第n隐藏层上增加神经元
	public void addNeuerInHidden(int n, int add_number, Fuction act_fuction){
		if(n< hidden_Neuron.length && n>=0 && add_number>0){
			hid_n[n] += add_number;
			Neuron[] newlist = new Neuron[hid_n[n]];
			
			int i;
			for(i=0; i< hidden_Neuron[n].length; i++)
				newlist[i] = hidden_Neuron[n][i];
			
			for(i= hidden_Neuron[n].length; i<hid_n[n]; i++)
			    newlist[i] = new Neuron(hidden_Neuron[n][0].w.length,act_fuction);
			
			//增加
			hidden_Neuron[n] = newlist;
			hiddenNeuer_out[n] = new float[hid_n[n]];
				
			//下一层
			if(n== hidden_Neuron.length-1)
				for(i=0;i<output_n;i++) output_Neuron[i].addInput_n(add_number);
			else
				for(i=0; i< hidden_Neuron[n+1].length; i++)
				     hidden_Neuron[n+1][i].addInput_n(add_number);
		}
	}
	public void addNeuerInHidden(int n,int add_number){
		addNeuerInHidden(n,add_number, Fuction.getFunctionById(act_fuctiom_ID));
	}

	//在第n隐藏层上减少神经元
	public void deleteNeuerInHidden(int n,int de_number){
		if(de_number>0 && n< hidden_Neuron.length && n>=0 && hid_n[n]>de_number ){
			hid_n[n] -= de_number;
			hidden_Neuron[n] = Arrays.copyOf(hidden_Neuron[n],hid_n[n]);
			hiddenNeuer_out[n] = new float[hid_n[n]];
			
			//下一层
			if(n== hidden_Neuron.length-1)
				for(int i=0;i<output_n;i++) output_Neuron[i].deleteInput_n(de_number);
			else
				for(int i = 0; i< hidden_Neuron[n+1].length; i++)
					hidden_Neuron[n+1][i].deleteInput_n(de_number);
		}
	}
	
	//在第n层前插入1层含有k个神经元的隐藏层
	public void addHiddenNeuer(int n, int k, Fuction act_fuction){
		if(n>=0 && k>0){
			Neuron[] newN = new Neuron[k];
			boolean flag = false;
			//初始化新加入的层
			if(n==0){
				for(int i=0;i<k;i++) newN[i] = new Neuron(input_n,act_fuction);
			}else if(flag = n>= hidden_Neuron.length){
				for(int i=0;i<k;i++) newN[i] = new Neuron(hid_n[hid_n.length-1],act_fuction);
			}else{
				for(int i=0;i<k;i++) newN[i] = new Neuron(hidden_Neuron[n-1].length,act_fuction);
			}
			
			//插入新层
			ArrayList<Neuron[]> newHiddenN = new ArrayList(Arrays.asList(hidden_Neuron));
			if(n<newHiddenN.size())
			    newHiddenN.add(n,newN);
			else
				newHiddenN.add(newN);

			int[] newHid_n = new int[newHiddenN.size()];
			float[][] newHidden_Out = new float[newHiddenN.size()][];
			for(int i=0;i<newHid_n.length;i++){
				newHid_n[i] = newHiddenN.get(i).length;
				newHidden_Out[i] = new float[newHid_n[i]];
			}
			
			//替换
			hidden_Neuron = newHiddenN.toArray(new Neuron[newHiddenN.size()][]);
			hiddenNeuer_out = newHidden_Out;
			hid_n = newHid_n;
			
			//修改下一层
			if(flag)
				for(int i = 0; i< output_Neuron.length; i++){
					output_Neuron[i].newInput_n(k);
				}
			else
				for(int i = 0; i< hidden_Neuron[n+1].length; i++){
					hidden_Neuron[n+1][i].newInput_n(k);
				}
				
		   System.out.println(Arrays.toString(hidden_Neuron[1]));
		}
	}
	public void addHiddenNeuer(int n,int k){
		addHiddenNeuer(n, k, Fuction.getFunctionById(act_fuctiom_ID));
	}

	//删除第n层神经元
	public void deletcHiddenNeuer(int n){
		if(n>=0 && n< hidden_Neuron.length && hidden_Neuron.length>1){
			ArrayList<Neuron[]> newHiddenN = new ArrayList(Arrays.asList(hidden_Neuron));
			//删除
			newHiddenN.remove(n);
			
			int[] newHid_n = new int[newHiddenN.size()];
			float[][] newHidden_Out = new float[newHiddenN.size()][];
			for(int i=0;i<newHid_n.length;i++){
				newHid_n[i] = newHiddenN.get(i).length;
				newHidden_Out[i] = new float[newHid_n[i]];
			}
			
			//替换
			hidden_Neuron = newHiddenN.toArray(new Neuron[newHiddenN.size()][]);
			hiddenNeuer_out = newHidden_Out;
			hid_n = newHid_n;
			
			//修改下一层
			if(n==0){
				for(int i=0;i<hid_n[0];i++)
					hidden_Neuron[0][i].newInput_n(input_n);
			}else if(n>=hid_n.length){
				for(int i = 0; i< output_Neuron.length; i++)
					output_Neuron[i].newInput_n(hid_n[hid_n.length-1]);
			}else{
				for(int i=0;i<hid_n[n];i++)
				    hidden_Neuron[n][i].newInput_n(hid_n[n-1]);
			}
		}
	}

    //把神经网络保存到文件
	public void saveInFile(String path) {
		File f = new File(path);
		if (f.exists()) {
			String p1 = path.substring(0, path.lastIndexOf("."));
			for (int i = 0; ; i++) {
				f = new File(path = p1 + "_" + i + ".log");
				if (!f.exists()) break;
			}
		}
		try {
			f.createNewFile();
		} catch (Exception e) {
		}

		if(f.isFile()) {
			FileWriter fw = null;
			try {
				fw = new FileWriter(f, true);
			} catch (IOException e) {
				e.printStackTrace();
			}
			PrintWriter pw = new PrintWriter(fw);

			pw.println("explain:" + EXPLAIN);
			pw.println(sInt("in_vector", input_n));
			pw.println(sInt("out_vector", output_n));
			pw.println(sFloat("nl", lr));
			pw.println(sIntArrays("hid_n", hid_n));
			pw.println(sFloat("de", dError));
			pw.println(sFloat("_in_max", _in_max));
			pw.println(sFloat("_out_max", _out_max));
			pw.println(sInt("ACT_FUCTION",act_fuctiom_ID));
			pw.println(sInt("LOSS_FUCTION", Loss_function.id));


			//输出层
			for (int i = 0; i < output_n; i++) {
				String name = "outputNeuer[" + i + "].";
				pw.println(sInt(name + "act_fuction_id", output_Neuron[i].ACT_function.id));
				pw.println(sFloat(name + "d", output_Neuron[i].b));
				for (int j = 0; j < output_Neuron[i].w.length; j++)
					pw.println(sFloat(name + "w" + j, output_Neuron[i].w[j]));
			}
			//隐藏层
			for (int n = 0; n < hidden_Neuron.length; n++)//第n层
				for (int i = 0; i < hidden_Neuron[n].length; i++) {//第i个
					String name = "hiddenNeuer[" + n + "][" + i + "].";
					pw.println(sInt(name + "act_fuction_id", hidden_Neuron[n][i].ACT_function.id));
					pw.println(sFloat(name + "d", hidden_Neuron[n][i].b));
					for (int j = 0; j < hidden_Neuron[n][i].w.length; j++)
						pw.println(sFloat(name + "w" + j, hidden_Neuron[n][i].w[j]));
				}

			pw.flush();
			try {
				fw.flush();
				pw.close();
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	//对象深复制
	@Override
    public Object clone() throws CloneNotSupportedException {
        BpNetwork p = (BpNetwork)super.clone();
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

	/****
	*****
	以下是私有函数
	*****
	*****/


	private float[] out_no_Save(float[] in){
		//计算输入层的输出
		int i;
		float[][] hiddenOut = new float[hiddenNeuer_out.length][];

		hiddenOut[0] = new float[hidden_Neuron[0].length];
		for( i = 0; i < hiddenOut[0].length; i++){
			hiddenOut[0][i] = hidden_Neuron[0][i].out_notSave(in);
		}

		//计算隐藏层输出
		for(i = 1; i < hidden_Neuron.length; i++){
			hiddenOut[i] = new float[hidden_Neuron[i].length];
			for( int j = 0; j < hiddenOut[i].length; j++ )
				hiddenOut[i][j] = hidden_Neuron[i][j].out_notSave(hiddenOut[i-1]);
		}

		float[] out = new float[output_n];
		//计算最终输出输
		for( i = 0; i < outNeuer_out.length; i++)
			out[i] = output_Neuron[i].out_notSave(hiddenOut[hiddenOut.length-1]);
		return out;
	}


	//神经网络输出
	private float[] out(float[] in){
		//计算输入层的输出
		int i;

		for(i=0; i < hidden_Neuron[0].length; i++){
		 	hiddenNeuer_out[0][i] = hidden_Neuron[0][i].out(in);
		}

		//计算隐藏层输出
		for(i = 1; i < hidden_Neuron.length; i++){
			for(int j = 0; j < hidden_Neuron[i].length; j++ )
				hiddenNeuer_out[i][j] = hidden_Neuron[i][j].out(hiddenNeuer_out[i-1]);
		}

		//计算最终输出输
		for(i=0;i<outNeuer_out.length;i++)
			outNeuer_out[i] = output_Neuron[i].out(hiddenNeuer_out[hiddenNeuer_out.length-1]);
		return outNeuer_out;
	}

	//动态选择线程数量
	private int getThread_n(int batch)  {
		float netSize = 0; //网络规模
		for(int i=1;i<hid_n.length;i++)
			netSize += hid_n[i-1]*hid_n[i];
		
		int n = (int) ((netSize*batch)/(1e5));
		
		//获取cpu核心数
		int core_number = Runtime.getRuntime().availableProcessors();
		
		if(n>core_number) n = core_number;
		
		return n;
	}


	//mini-batch
	private void upgrade_mini_batch(float[][] in,float[][] t,int batch_size,int thread_n){
		float[][] in_batch = new float[batch_size][];
		float[][] t_batch = new float[batch_size][];
		for(int j=0;j<in_batch.length;j++){
			//随机选出n个作为mini-batch
			int index = (int)(Math.random()*in.length);
			in_batch[j] = in[index];
			t_batch[j] = t[index];
		}
		upgrade(in_batch,t_batch,thread_n);
	}


/*
	private class TrainData {
		public float[] train_x;
		public float[] train_y;
		public TrainData(float[] train_x, float[] train_y){
			this.train_x = train_x;
			this.train_y = train_y;
		}

		@Override
		public String toString() {
			return "TrainData{" +
					"train_x=" + Arrays.toString(train_x) +
					", train_y=" + Arrays.toString(train_y) +
					'}';
		}
	}


	public  ArrayList<float[][]>[]  splitBatch(float[][] in, float[][] t, int batch_size){
		TrainData[] trainDatas = new TrainData[in.length];
		for (int i = 0; i < trainDatas.length; i++)
			trainDatas[i] = new TrainData(in[i], t[i]);

		ArrayList<TrainData> d0 = new ArrayList<>();
		Collections.addAll(d0, trainDatas);

		//随机打乱
		Collections.shuffle(d0);
		//System.out.println("splitBatch(): d0.size=" + d0.size() + "   " + d0.get(in.length - 1));

		ArrayList<TrainData[]> trainDataList = new ArrayList<>();

		TrainData[] var0 = new TrainData[batch_size];
		for (int i = 0; i < d0.size(); i++){
			TrainData ti = d0.get(i);
			if( i%batch_size == 0 && i>0){
				trainDataList.add(var0);

				int n = d0.size() - i;
				if(n > batch_size) n = batch_size;
				var0 = new TrainData[n];
			}
			var0[i%batch_size] = ti;
		}

		//System.out.println("splitBatch(): trainDataList.size=" + trainDataList.size());


		if(in.length % batch_size > 16){
			trainDataList.add(var0);
			//System.out.println(trainDataList.get(trainDataList.size()-1).length + "   " + in.length % batch_size);
		}

		ArrayList<float[][]> train_x, train_y;
		train_x = new ArrayList<>();
		train_y = new ArrayList<>();
		for (TrainData[] di : trainDataList){
			//System.out.println(di);
			float[][] x = new float[di.length][];
			float[][] y = new float[di.length][];
			for(int i = 0; i < di.length; i++){

				x[i] = di[i].train_x;
				y[i] = di[i].train_y;
			}
			train_x.add(x);
			train_y.add(y);
		}

		//System.out.println("splitBatch(): train_x: " + train_x.size() + "    " + Arrays.toString(train_x.get(0)));
		//System.out.println("splitBatch(): train_y: " + train_y.size() + "    " + Arrays.toString(train_y.get(0)));

		ArrayList[] data = new ArrayList[2];
		data[0] = train_x;
		data[1] = train_y;
		return data;
	}

 */


	/*
	//批量梯度下降更新thread_n:开启的线程数
	private void upgrade(float[][] train_X, float[][] train_Y, int ThreadNumber){
		int len = train_X.length;
		int i;
		if(ThreadNumber < 2){
			//单线程
		    for(int j = 0; j < len; j++) {
			 	upgradeBatch(train_X[j], train_Y[j]);//耗时点
			}
		}else{
		    //多线程运算
		    Thread_upgrade upthread = new Thread_upgrade(train_X,train_Y);
		    Future[] futureList = new Future[ThreadNumber];

		    for(i=0; i<ThreadNumber ;i++) {
				futureList[i] = upThreadPool.submit(new Thread(upthread));
			}

			try {
				for(Future future: futureList) future.get();
			}catch (InterruptedException e){
				e.printStackTrace();
			}catch(ExecutionException e){
				e.printStackTrace();
			}
		}


		//更新权重
		for (i = 0; i < output_Neuer.length; i++) {//star for
				for (int j = 0; j < output_Neuer[i].w.length; j++) {
					//优化器
					output_Neuer[i].data_list[j] = deltaOptimizer.DELTA(output_Neuer[i].data_list[j] / len);

					output_Neuer[i].setW(j, output_Neuer[i].w[j] + lr * output_Neuer[i].data_list[j]);
					output_Neuer[i].data_list[j] = 0;
				}

			    output_Neuer[i].data = deltaOptimizer.DELTA(output_Neuer[i].data / len);

				output_Neuer[i].d += lr * output_Neuer[i].data;
				output_Neuer[i].data = 0;
		}//end for


		//隐藏层
		for(i=hidden_Neuer.length-1;i>0;i--)
			for (int j = 0; j < hidden_Neuer[i].length; j++) {
				for (int k = 0; k < hidden_Neuer[i][j].w.length; k++) {

					    hidden_Neuer[i][j].data_list[k] = deltaOptimizer.DELTA(hidden_Neuer[i][j].data_list[k] / len);

						hidden_Neuer[i][j].setW(k, hidden_Neuer[i][j].w[k] + lr * hidden_Neuer[i][j].data_list[k]);
						hidden_Neuer[i][j].data_list[k] = 0;
				}//end for


				hidden_Neuer[i][j].data = deltaOptimizer.DELTA(hidden_Neuer[i][j].data /len);

				hidden_Neuer[i][j].d += lr * hidden_Neuer[i][j].data;
				hidden_Neuer[i][j].data = 0;
			}//end for


	    if(hidden_Neuer.length>1)//和输入层连接的隐藏层
	    	for (i = 0; i < hidden_Neuer[0].length; i++) {
	    		for (int j = 0; j < hidden_Neuer[0][i].w.length; j++) {//更新

					hidden_Neuer[0][i].data_list[j] = deltaOptimizer.DELTA(hidden_Neuer[0][i].data_list[j] / len);

	    			hidden_Neuer[0][i].setW(j, hidden_Neuer[0][i].w[j] + lr * hidden_Neuer[0][i].data_list[j] );
	    			hidden_Neuer[0][i].data_list[j] = 0;
	    		}

				hidden_Neuer[0][i].data = deltaOptimizer.DELTA(hidden_Neuer[0][i].data / len);

	    		hidden_Neuer[0][i].d += lr * hidden_Neuer[0][i].data;
	    		hidden_Neuer[0][i].data = 0;
	    	}//end for

	}

	 */


	private void UpdateWeights(UpgradeThreadData[] upgradeThreadDates, int threadNumber){
		//final int b = 2;

		final int batch_size = upgradeThreadDates.length;
		final int wn = upgradeThreadDates[0].getWn();

		UpgradeThreadData upd = upgradeThreadDates[0];

		final int var0 = upd.outNeuer_delta_w.length * upd.outNeuer_delta_w[0].length;
		final int[] hVar = new int[upd.hiddenNeuer_delta_w.length];
		final int[] bVar = new int[upd.hiddenNeuer_delta_w.length];

		for (int i = 0; i < hVar.length; i++) {
			hVar[i] = upd.hiddenNeuer_delta_w[i].length * upd.hiddenNeuer_delta_w[i][0].length;
			if( i == 0 ) bVar[i] = 0; else bVar[i] = bVar[i-1] + hid_n[i];
		}

		int splitN = 65536;
		for (int index = 0; index < wn; index += splitN) {
			int workN = splitN;
			if(index + splitN > wn) workN = wn - index;
			ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(workN) {

				@Override
				public void working(int index) {

					float delta_w = 0;
					if (index < var0) { //output layer
						int p1 = index / upd.outNeuer_delta_w[0].length;
						int p2 = index % upd.outNeuer_delta_w[0].length;

						if (p2 == 0) { //同时更新b
							float delta_b = 0;
							for (UpgradeThreadData upgradeThreadDate : upgradeThreadDates) {
								delta_b += upgradeThreadDate.outNeuer_delta[p1];
								delta_w += upgradeThreadDate.outNeuer_delta_w[p1][p2];
							}

							delta_b /= batch_size;

							int p = wn + p1;
							delta_b = deltaOptimizer.DELTA(delta_b, p);

							output_Neuron[p1].b += lr * delta_b;

						} else {
							for (UpgradeThreadData upgradeThreadDate : upgradeThreadDates)
								delta_w += upgradeThreadDate.outNeuer_delta_w[p1][p2];

						}

						delta_w /= batch_size;

						delta_w = deltaOptimizer.DELTA(delta_w, index);
						output_Neuron[p1].setW(p2, output_Neuron[p1].w[p2] + lr * delta_w);

					} else { //hidden layer

						int ap = index;

						index -= var0;
						int p0 = 0, p1, p2;

						int h = 0;
						for (int hwn : hVar) {
							h += hwn;
							if (index < h) break;
							p0++;
						}

						h -= hVar[p0];
						index -= h;

						p1 = index / upd.hiddenNeuer_delta_w[p0][0].length;
						p2 = index % upd.hiddenNeuer_delta_w[p0][0].length;

						if (p1 == 0) { //同时更新b
							float delta_b = 0;
							for (UpgradeThreadData upgradeThreadDate : upgradeThreadDates) {
								delta_b += upgradeThreadDate.hiddenNeuer_delta[p0][p1];
								delta_w += upgradeThreadDate.hiddenNeuer_delta_w[p0][p1][p2];
							}
							delta_b /= -batch_size;

							int p = wn + upd.outNeuer_delta.length + bVar[p0] + p1;
							delta_b = deltaOptimizer.DELTA(delta_b, p);

							hidden_Neuron[p0][p1].b += lr * delta_b;

						} else {
							for (UpgradeThreadData upgradeThreadDate : upgradeThreadDates)
								delta_w += upgradeThreadDate.hiddenNeuer_delta_w[p0][p1][p2];

						}

						delta_w /= batch_size;

						delta_w = deltaOptimizer.DELTA(delta_w, ap);

						hidden_Neuron[p0][p1].setW(p2, hidden_Neuron[p0][p1].w[p2] + lr * delta_w);

					}
				}

			};
			threadWorker.setStart_index(index);
			ThreadWork.start(threadWorker, threadNumber);
		}

	}

	/*
	private void UpdateWeights0(UpgradeThreadData[] upgradeThreadDates) {
		int len = upgradeThreadDates.length;
		int i;



		//更新权重
		for (i = 0; i < output_Neuer.length; i++) {//star for
			for (int j = 0; j < output_Neuer[i].w.length; j++) {

				float delta_w = 0;
				for(int k = 0; k < len; k++)
					delta_w += upgradeThreadDates[k].outNeuer_delta_w[i][j];

				delta_w /= len;

				//优化器
				delta_w = deltaOptimizer.DELTA(delta_w,0);

				output_Neuer[i].setW(j, output_Neuer[i].w[j] + lr * delta_w);
				//output_Neuer[i].data_list[j] = 0;
			}

			float delta = 0;
			for(int k = 0; k < len; k++)
				delta -= upgradeThreadDates[k].outNeuer_delta[i];

			delta = ( delta / len );
			delta = deltaOptimizer.DELTA( delta,0 );

			output_Neuer[i].d += lr * delta;
			//output_Neuer[i].data = 0;
		}//end for


		//隐藏层
		for(i=hidden_Neuer.length-1;i>0;i--)
			for (int j = 0; j < hidden_Neuer[i].length; j++) {
				for (int k = 0; k < hidden_Neuer[i][j].w.length; k++) {

					float delta_w = 0;
					for(int l = 0; l < len; l++ )
						delta_w += upgradeThreadDates[l].hiddenNeuer_delta_w[i][j][k];

					delta_w /= len;

					delta_w = deltaOptimizer.DELTA(delta_w,0);

					hidden_Neuer[i][j].setW(k, hidden_Neuer[i][j].w[k] + lr * delta_w);
					//hidden_Neuer[i][j].data_list[k] = 0;
				}//end for

				float delta = 0;
				for(int k = 0; k < len; k++)
					delta -= upgradeThreadDates[k].hiddenNeuer_delta[i][j];

				delta =  ( delta / len );

				delta = deltaOptimizer.DELTA(delta,0);

				hidden_Neuer[i][j].d += lr * delta;
				//hidden_Neuer[i][j].data = 0;
			}//end for


		if(hidden_Neuer.length>1)//和输入层连接的隐藏层
			for (i = 0; i < hidden_Neuer[0].length; i++) {
				for (int j = 0; j < hidden_Neuer[0][i].w.length; j++) {//更新

					float delta_w = 0;
					for(int k = 0; k < len; k++)
						delta_w += upgradeThreadDates[k].hiddenNeuer_delta_w[0][i][j];

					delta_w /= len;

					delta_w = deltaOptimizer.DELTA(delta_w,0);

					hidden_Neuer[0][i].setW(j, hidden_Neuer[0][i].w[j] + lr * delta_w );
					//hidden_Neuer[0][i].data_list[j] = 0;
				}

				float delta = 0;
				for (int j = 0; j < len; j++)
					delta -= upgradeThreadDates[j].hiddenNeuer_delta[0][i];

				delta =  ( delta / len );

				delta = deltaOptimizer.DELTA(delta,0);

				hidden_Neuer[0][i].d += lr * delta;
				//hidden_Neuer[0][i].data = 0;
			}//end for
	}
	 */


	// batch
	public void upgrade(float[][] train_X, float[][] train_Y, int threadNumber){
		//System.out.println("upgrade(): init" );
		final UpgradeThreadData[] upgradeThreadData = new UpgradeThreadData[train_X.length];
		ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(train_X.length){
			@Override
			public void working(int index) {
				upgradeThreadData[index] = new UpgradeThreadData();
				calculateDelta(train_X[index], train_Y[index], upgradeThreadData[index]);
			}
			@Override
			public Object getObject() {
				return upgradeThreadData;
			}
		};

		ThreadWork.start(threadWorker, threadNumber);
		UpdateWeights(upgradeThreadData, threadNumber);
	}


	/*
	//随机梯度下降更新，batch-size = 1;
	public void upgrade(float[] input,float[] target){
		float[] out = out(input);
		int i;
		//输出层
		for(i=0;i<output_Neuer.length;i++){//star for

			if(output_Neuer[i].ACT_function.id==1 && Loss_function.id==11)
				 output_Neuer[i].delta = -(out[i] - target[i]);
			else
			     output_Neuer[i].delta = -Loss_function.f_derivate(out[i],target[i]) * output_Neuer[i].ACT_function.f_derivate(output_Neuer[i].lastOut);

			float x1 = output_Neuer[i].delta;

 			for(int j=0;j<output_Neuer[i].w.length;j++) {

				output_Neuer[i].setW(j, output_Neuer[i].w[j] + lr * deltaOptimizer.DELTA(x1 * hiddenNeuer_out[hidden_Neuer.length - 1][j], 0));//= tnl * deltaOptimizer.DELTA( x1 * hiddenNeuer_out[hidden_Neuer.length-1][j]) );

			}

			output_Neuer[i].d += lr * deltaOptimizer.DELTA(-x1, 0);
		}//end for

		float delta;
		int n;
		//隐藏层
		for(i=hidden_Neuer.length-1;i>0;i--){
			for(int j=0;j<hidden_Neuer[i].length;j++){
				delta = 0;
				if(i == hidden_Neuer.length-1){
					n = output_Neuer.length;
					for(int k=0; k<n; k++)
						delta += output_Neuer[k].delta * output_Neuer[k].w_old[j];
				}
				else{
					n = hidden_Neuer[i+1].length;
					for(int k=0; k<n; k++)
						delta += hidden_Neuer[i+1][k].delta * hidden_Neuer[i+1][k].w_old[j];
				}// end if

				delta *= hidden_Neuer[i][j].ACT_function.f_derivate(hidden_Neuer[i][j].lastOut);

				//更新权值和阀值
				for (int k = 0; k < hidden_Neuer[i][j].w.length; k++)
					hidden_Neuer[i][j].setW(k,hidden_Neuer[i][j].w[k] + lr * deltaOptimizer.DELTA(delta * hiddenNeuer_out[i - 1][k], 0));

				hidden_Neuer[i][j].d += lr * deltaOptimizer.DELTA(-delta, 0);

			}


		}//end for


		if(hidden_Neuer.length>1)//和输入层连接的隐藏层
			for(i=0;i<hidden_Neuer[0].length;i++){
				 delta = 0;
				for(int j=0;j<hidden_Neuer[1].length;j++)
					delta += hidden_Neuer[1][j].delta * hidden_Neuer[1][j].w_old[i];

				delta *= hidden_Neuer[0][i].ACT_function.f_derivate(hidden_Neuer[0][i].lastOut);
				hidden_Neuer[0][i].delta = delta;

				for(int j=0;j<hidden_Neuer[0][i].w.length;j++)//更新
					hidden_Neuer[0][i].setW(j,hidden_Neuer[0][i].w[j] + lr * deltaOptimizer.DELTA(delta * input[j], 0));

				hidden_Neuer[0][i].d += lr * deltaOptimizer.DELTA(-delta, 0);
			}//end for

	}// end upgrade

	 */

	/*
	//批量梯度下降，多线程并行计算
	private class Thread_upgrade implements Runnable{
		private boolean finish = false;
		private float[][] in;
		private float[][] t;
		private boolean[] nfinish;//是否计算完成
		private boolean[] lock;//是否锁了(正在计算)
		
		public Thread_upgrade(float[][] in_,float[][] t_){
			in = in_;
			t = t_;
			nfinish = new boolean[in.length];
			lock = new boolean[in.length];
		}
		
		public void run(){
			UpgradeThreadData td = new UpgradeThreadData();
			while(true){
				if(finish) break;
				boolean flag = true;
				for(int p=0;p<in.length;p++)
					if(!nfinish[p]){	//第i个未完成
						flag = false; 
						if(!lock[p]){//i个未锁
							lock[p] = true; //上锁
						    //doing
							calculateDelta(in[p], t[p], td);
							//处理完成,
							nfinish[p] = true;
						}
				    }	  
				if(flag){
					finish = true;//已完成
					break;
				}
			}//end while
		}//end run
	}

 */
	private class UpgradeThreadData {//数据复用

		float[] out = new float[outNeuer_out.length];
		float[] outNeuer_delta = new float[output_Neuron.length];
		float[][] outNeuer_delta_w = new float[output_Neuron.length][output_Neuron[0].w.length];

		//隐藏层每个输出
		float[][] hiddenNeuer_out_ = new float[hidden_Neuron.length][];
		//隐藏层每个delta
		float[][] hiddenNeuer_delta = new float[hidden_Neuron.length][];
		float[][][] hiddenNeuer_delta_w = new float[hidden_Neuron.length][][];

        
		public UpgradeThreadData(){
		    for(int i = 0; i < hid_n.length; i++){//for
			   hiddenNeuer_out_[i] = new float[hid_n[i]];
			   hiddenNeuer_delta[i] = new float[hid_n[i]];
			   hiddenNeuer_delta_w[i] = new float[hid_n[i]][hidden_Neuron[i][0].w.length];
		  }
		}// end for

		public int getWn(){
			//[ (outNeuer_delta_w ), (hiddenNeuer_delta_w[0], hiddenNeuer_delta_w[1].....)]
			int wn = outNeuer_delta_w.length * outNeuer_delta_w[0].length;
			for (float[][] hw: hiddenNeuer_delta_w)
				wn += hw.length * hw[0].length;
			return wn;
		}
	}


	//计算梯度值
	private void calculateDelta(float[] in, float[] t, UpgradeThreadData upgradeThreadData){
		int i;
		for(int j = 0; j < hidden_Neuron[0].length; j++)
				upgradeThreadData.hiddenNeuer_out_[0][j] = hidden_Neuron[0][j].act_f(hidden_Neuron[0][j].LastIn_notSave(in));

		//计算隐藏层输出
		for(i = 1; i < hidden_Neuron.length; i++ )
			for(int j = 0; j< hidden_Neuron[i].length; j++)
				upgradeThreadData.hiddenNeuer_out_[i][j] = hidden_Neuron[i][j].act_f(hidden_Neuron[i][j].LastIn_notSave(upgradeThreadData.hiddenNeuer_out_[i-1]));


		//计算最终输出输
		for( i = 0; i < outNeuer_out.length; i++ )
			upgradeThreadData.out[i] = output_Neuron[i].act_f(output_Neuron[i].LastIn_notSave(upgradeThreadData.hiddenNeuer_out_[upgradeThreadData.hiddenNeuer_out_.length-1]));


		float delta;
		//以下计算梯度
		for(i = 0; i < output_Neuron.length; i++){//star for
			if(output_Neuron[i].ACT_function.id==1 && Loss_function.id==11) {
				//System.out.println(upgradeThreadData.out[i]);
				delta = -(upgradeThreadData.out[i] - t[i]);
			}else
			     delta = -Loss_function.f_derivative(upgradeThreadData.out[i],t[i]) * output_Neuron[i].ACT_function.f_derivative(upgradeThreadData.out[i]);

			upgradeThreadData.outNeuer_delta[i] = delta;
			for(int j = 0; j < output_Neuron[i].w.length; j++) {
				//output_Neuer[i].data_list[j] += delta * upgradeThreadData.hiddenNeuer_out_[hidden_Neuer.length - 1][j];
				upgradeThreadData.outNeuer_delta_w[i][j] = delta * upgradeThreadData.hiddenNeuer_out_[hidden_Neuron.length - 1][j];
			}

			//output_Neuer[i].data += -delta;
		}//end for

		int n;
		//隐藏层
		for(i= hidden_Neuron.length-1; i>0; i--){
			for(int j = 0; j< hidden_Neuron[i].length; j++){
				delta = 0;
				if(i == hidden_Neuron.length-1){
					n = output_Neuron.length;
					for(int k=0; k<n; k++)
						delta += upgradeThreadData.outNeuer_delta[k] * output_Neuron[k].w[j];//w_old[j];
				}
				else{
					n = hidden_Neuron[i+1].length;
					for(int k=0; k<n; k++)
						delta += upgradeThreadData.hiddenNeuer_delta[i+1][k] * hidden_Neuron[i+1][k].w[j];//w_old[j];
				}// end if

				delta *= hidden_Neuron[i][j].ACT_function.f_derivative(upgradeThreadData.hiddenNeuer_out_[i][j]);
				upgradeThreadData.hiddenNeuer_delta[i][j] = delta;
				//计算更新权值和阀值0
				for(int k = 0; k< hidden_Neuron[i][j].w.length; k++) {
					//hidden_Neuer[i][j].data_list[k] += delta * upgradeThreadData.hiddenNeuer_out_[i - 1][k];
					upgradeThreadData.hiddenNeuer_delta_w[i][j][k] = delta * upgradeThreadData.hiddenNeuer_out_[i - 1][k];
				}

				//hidden_Neuer[i][j].data += -delta;
			}
		}//end for

		if(hidden_Neuron.length>1)//和输入层连接的隐藏层
			for(i=0; i< hidden_Neuron[0].length; i++){
				delta = 0;
				for(int j = 0; j< hidden_Neuron[1].length; j++)
					delta += upgradeThreadData.hiddenNeuer_delta[1][j] * hidden_Neuron[1][j].w[i];//w_old[i];

				delta *= hidden_Neuron[0][i].ACT_function.f_derivative(upgradeThreadData.hiddenNeuer_out_[0][i]);
				upgradeThreadData.hiddenNeuer_delta[0][i] = delta;

				for(int j = 0; j< hidden_Neuron[0][i].w.length; j++) {//更新
					//hidden_Neuer[0][i].data_list[j] += delta * in[j];
					upgradeThreadData.hiddenNeuer_delta_w[0][i][j] =  delta * in[j];
				}

				//hidden_Neuer[0][i].data += -delta;
			}//end for

	}


	//batch-size>1	批量梯度下降，单线程
	/*
	private float upgradeBatch(float[] input, float[] target){
		//System.out.println("批量梯度下降，单线程");
		float[] out = out(input);
		//System.out.println("out " + Arrays.toString(out));
		int i;
		//输出层
		for(i=0; i< output_Neuron.length; i++){//star for

			if(output_Neuron[i].ACT_function.id==1 && Loss_function.id==11)
				output_Neuron[i].delta = -(out[i] - target[i]);
			else
				output_Neuron[i].delta = -Loss_function.f_derivative(out[i],target[i]) * output_Neuron[i].ACT_function.f_derivative(output_Neuron[i].lastOut);

			float x1 = output_Neuron[i].delta;//mmmmmmmm
			//System.out.print("  hd2: " + -Loss_function.f_derivate(out[i],target[i]));
			for(int j = 0; j< output_Neuron[i].w.length; j++)
				output_Neuron[i].data_list[j] += x1 * hiddenNeuer_out[hidden_Neuron.length-1][j];

			output_Neuron[i].data += -x1;
		}//end for

		float delta;
		//隐藏层
		for(i= hidden_Neuron.length-1; i>0; i--){
			for(int j = 0; j< hidden_Neuron[i].length; j++){
			    delta = 0;
				int n;
				if(i == hidden_Neuron.length-1){
					n = output_Neuron.length;
				    for(int k=0; k<n; k++)
						delta += output_Neuron[k].delta * output_Neuron[k].w[j];
				}
				else{
					n = hidden_Neuron[i+1].length;
				    for(int k=0; k<n; k++)
					    delta += hidden_Neuron[i+1][k].delta * hidden_Neuron[i+1][k].w[j];
				}// end if
				delta *= hidden_Neuron[i][j].ACT_function.f_derivative(hidden_Neuron[i][j].lastOut);
				hidden_Neuron[i][j].delta = delta;


				//计算更新权值和阀值
				for(int k = 0; k< hidden_Neuron[i][j].w.length; k++)
					hidden_Neuron[i][j].data_list[k] += delta * hiddenNeuer_out[i-1][k];

				hidden_Neuron[i][j].data += -delta;
			}
		}//end for

	    if(hidden_Neuron.length>1)//和输入层连接的隐藏层
			for(i=0; i< hidden_Neuron[0].length; i++){
				delta = 0;

				for(int j = 0; j< hidden_Neuron[1].length; j++)
					delta += hidden_Neuron[1][j].delta * hidden_Neuron[1][j].w[i];

				delta *= hidden_Neuron[0][i].ACT_function.f_derivative(hidden_Neuron[0][i].lastOut);
				hidden_Neuron[0][i].delta = delta;

				for(int j = 0; j< hidden_Neuron[0][i].w.length; j++)//更新
					hidden_Neuron[0][i].data_list[j] += delta * input[j];

				hidden_Neuron[0][i].data += -delta;
			}//end for

		return Loss(out,target);
	}// end upgrade
	 */

	
	//初始化
	private void init(Fuction act_fuction){
		Loss_function = new MSELoss();

		output_Neuron = new Neuron[output_n];
		outNeuer_out = new float[output_n];
		for(int i = 0; i< output_Neuron.length; i++)
			output_Neuron[i] = new Neuron(hid_n[hid_n.length-1],act_fuction);

		hidden_Neuron = new Neuron[hid_n.length][];
		hiddenNeuer_out = new float[hid_n.length][];
		for(int i=0;i<hid_n.length;i++){//for
			Neuron[] n =new Neuron[hid_n[i]];
			hiddenNeuer_out[i] = new float[hid_n[i]];
			if(i==0)
				for(int j=0;j<n.length;j++) 
					n[j] = new Neuron(input_n,act_fuction);
			else 
				for(int j=0;j<n.length;j++)
					n[j] = new Neuron(hid_n[i-1],act_fuction);

			hidden_Neuron[i] = n;
		}// end for

	}

	//损失函数
	private float Loss(float[] o, float[] t){
		float r = 0;
		for(int i=0; i<o.length; i++){
			r += Loss_function.f(o[i],t[i]);
		}
		return r / t.length;
	}


	//获取二维数组中的最大值
	private float getListMax(float[][] data){
		/*
		float[] _data = new float[data.length];

		for(int i=0;i<data.length;i++){
			DoubleSummaryStatistics stat = Arrays.stream(data[i]).summaryStatistics();
			// float min = stat.getMin();
			float max = stat.getMax();
			_data[i] = max;
		}

		DoubleSummaryStatistics stat2 = Arrays.stream(_data).summaryStatistics();

		return stat2.getMax();*/

		float max = -9999999.0f;
		for (float[] datum : data)
			for (float v : datum) {
				if (v > max)
					max = v;
			}
		return max;
	}

	//数组连接
	private static float[] merge(float[] a1, float[] a2) {
		float[] a3 = new float[a1.length + a2.length];
		System.arraycopy(a1, 0, a3, 0, a1.length);
		System.arraycopy(a2, 0, a3, a1.length, a2.length);
		return a3;
	}



	//从文件加载神经网络
	private void readFile(String path){
		File file = new File(path);
		if(file.isFile())
			try {
				FileReader fileReader = null;
				fileReader = new FileReader(file);
				BufferedReader in = new BufferedReader(fileReader);
				String line = in.readLine();
				EXPLAIN = line.substring(8);
				line = in.readLine();
				input_n = getSInt(line);
				line = in.readLine();
				output_n = getSInt(line);
				line = in.readLine();
				lr = getSFloat(line);
				line = in.readLine();
				hid_n = getsIntArrays(line);
				line = in.readLine();
				dError = getSFloat(line);
				line = in.readLine();
				_in_max = getSFloat(line);
				line = in.readLine();
				_out_max = getSFloat(line);
				line = in.readLine();
				act_fuctiom_ID = getSInt(line);
				line = in.readLine();
				Loss_function = Fuction.getFunctionById(getSInt(line));


				outNeuer_out = new float[output_n];
				hiddenNeuer_out = new float[hid_n.length][];
				for(int i=0;i<hid_n.length;i++)
					hiddenNeuer_out[i] = new float[hid_n[i]];



				output_Neuron = new Neuron[output_n];
				for(int i=0;i<output_n;i++) {
					line = in.readLine();
					Fuction actf = Fuction.getFunctionById(getSInt(line));
					line = in.readLine();
					float d = getSFloat(line);
					float[] w = new float[hid_n[hid_n.length-1]];
					for(int j=0;j<w.length;j++){
						line = in.readLine();
						w[j] = getSFloat(line);
					}

					Neuron neuer = new Neuron(w.length);
					neuer.w = w;
					neuer.b = d;
					neuer.ACT_function = actf;
					output_Neuron[i] = neuer;
				}

				hidden_Neuron = new Neuron[hid_n.length][];
				for (int i=0;i<hid_n.length;i++)
					hidden_Neuron[i] = new Neuron[hid_n[i]];
				//第0层隐藏层
				for(int i = 0; i< hidden_Neuron[0].length; i++){
					line = in.readLine();
					Fuction actf = Fuction.getFunctionById(getSInt(line));
					line = in.readLine();
					float d = getSFloat(line);
					float[] w = new float[input_n];
					for(int j=0;j<w.length;j++){
						line = in.readLine();
						w[j] = getSFloat(line);
					}
					Neuron neuer = new Neuron(w.length);
					neuer.w = w;
					neuer.b = d;
					neuer.ACT_function = actf;
					hidden_Neuron[0][i] = neuer;
				}

				for(int n = 1; n< hidden_Neuron.length; n++)//第n层
					for(int i = 0; i< hidden_Neuron[n].length; i++){
						line = in.readLine();
						Fuction actf = Fuction.getFunctionById(getSInt(line));
						line = in.readLine();
						float d = getSFloat(line);
						float[] w = new float[hid_n[n-1]];
						for(int j=0;j<w.length;j++){
							line = in.readLine();
							w[j] = getSFloat(line);
						}
						Neuron neuer = new Neuron(w.length);
						neuer.w = w;
						neuer.b = d;
						neuer.ACT_function = actf;
						hidden_Neuron[n][i] = neuer;
					}

		        in.close();
		        fileReader.close();
	       }catch(Exception e){ System.out.println(e); }
    }




	//获取神经网络的参数数量
	public int getWandD_number(){
		//输入层
		int r = 0;//input_Neuer.length + input_Neuer.length * input_Neuer[0].w.length;

		//隐藏
		for (Neuron[] neuers : hidden_Neuron) {
			//d,                    w
			r += neuers.length + neuers.length * neuers[0].w.length;
		}
		r += output_Neuron.length + output_Neuron.length * output_Neuron[0].w.length;
		return r;
	}
}




