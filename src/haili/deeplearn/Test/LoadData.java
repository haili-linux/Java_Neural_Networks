package haili.deeplearn.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class LoadData
{
    /*public static void main(String[] args){
		int[] arf = new int[]{800,800,100};
		//BPNetwork bpNetwork = new BPNetwork(2,1,0.1,arf);
		BPNetwork bpNetwork = new BPNetwork("F:\\java Project\\圣遗物强化分析\\保存的神经网络\\" + "1.txt");
		//bpNetwork.saveInFile("F:\\java Project\\圣遗物强化分析\\保存的神经网络\\1.txt");
		System.out.println("sss");
		System.out.println(bpNetwork.out_(new float[]{0.22,0.81})[0]);
		//0.43001386252443763

	}*/
	//input:(mc1,fc1_1,fc1_2,fc1_3,fc1_4,rc1,rn1,mc2,fc2_1,...,mc10,fc10_1,fc10_2..,rc10)
	//float[] B[],b[];
	//one-hot
    final static String ct = "BbjcGgSsFf";
	final static float[] B = {1,0,0,0,0,0,0,0,0,0};
	final static float[] b = {0,1,0,0,0,0,0,0,0,0};
	final static float[] j = {0,0,1,0,0,0,0,0,0,0};
	final static float[] c = {0,0,0,1,0,0,0,0,0,0};
	final static float[] G = {0,0,0,0,1,0,0,0,0,0};
	final static float[] g = {0,0,0,0,0,1,0,0,0,0};
	final static float[] S = {0,0,0,0,0,0,1,0,0,0};
	final static float[] s = {0,0,0,0,0,0,0,1,0,0};
	final static float[] F = {0,0,0,0,0,0,0,0,1,0};
	final static float[] f = {0,0,0,0,0,0,0,0,0,1};
	final static float[] y = {0,0,0,0,0,0,0,0,0,0};
	
	final static String B_str = "1000000000";
	final static String b_str = "0100000000";
	final static String j_str = "0010000000";
	final static String c_str = "0001000000";
	final static String G_str = "0000100000";
	final static String g_str = "0000010000";
	final static String S_str = "0000001000";
	final static String s_str = "0000000100";
	final static String F_str = "0000000010";
	final static String f_str = "0000000001";
	final static String y_str = "0000000000";
	//档位
	final static float[] n1 = {1,0,0,0};
	final static float[] n2 = {0,1,0,0};
	final static float[] n3 = {0,0,1,0};
	final static float[] n4 = {0,0,0,1};
	final static String n1_str = "1000";
	final static String n2_str = "0100";
	final static String n3_str = "0010";
	final static String n4_str = "0001";

	//倍数
	final static float[] b0 = {0, 0};
	final static float[] b2 = {1, 0};
	final static float[] b5 = {0, 1};
	final static String b0_str = "00";
	final static String b2_str = "10";
	final static String b5_str = "01";
	
	
	/*u1:
	每次强化记录编码为64位二进制数，1-10位为主词条，
	11-20位为副词条1,21-30位为副词条2,31-40位是副词条3
    41-50位为副词条4,51-60位是强化结果,61-64是结果档位*/
	public static String u1(String data){
		String rdata = "";
		//主
		String mainc = data.substring(0,1);

		int index1 = data.indexOf("(") +1;
		int index2 = data.indexOf(")");
		//副
		String fc = data.substring(index1,index2);
		
		int index3 = index2+3;
		String rc = data.substring(index3,index3+2);
		
		//结果档位
		String rn = rc.substring(1);
		//结果
		rc = rc.substring(0,1);
		
		switch(mainc){
			case "B": rdata += B_str; break;
			case "b": rdata += b_str; break;
			case "j": rdata += j_str; break;
			case "c": rdata += c_str; break;
			case "G": rdata += G_str; break;
			case "g": rdata += g_str; break;
			case "S": rdata += S_str; break;
			case "s": rdata += s_str; break;
			case "F": rdata += F_str; break;
			case "y": rdata += y_str; break;
		}


		int fn = fc.length();
		for(int i=0;i<4;i++){
		    String a; 
			if(i<fn) 
				a = fc.charAt(i)+"";
			else
				a = "";
			switch(a){
				case "B": rdata += B_str; break;
				case "b": rdata += b_str; break;
				case "j": rdata += j_str; break;
				case "c": rdata += c_str; break;
				case "G": rdata += G_str; break;
				case "g": rdata += g_str; break;
				case "S": rdata += S_str; break;
				case "s": rdata += s_str; break;
				case "F": rdata += F_str; break;
			    case "f": rdata += f_str; break;
				default:  rdata += y_str; break;
			}
		}


		switch(rc){
			case "B": rdata += B_str; break;
			case "b": rdata += b_str; break;
			case "j": rdata += j_str; break;
			case "c": rdata += c_str; break;
			case "G": rdata += G_str; break;
			case "g": rdata += g_str; break;
			case "S": rdata += S_str; break;
			case "s": rdata += s_str; break;
			case "F": rdata += F_str; break;
			case "f": rdata += f_str; break;
			default: break;
		}
		
		switch(rn){
			case "1": rdata += n1_str; break;
			case "2": rdata += n2_str; break;
			case "3": rdata += n3_str; break;
			case "4": rdata += n4_str; break;
			default: break;
		}
		return rdata;
	}
	public static String u1__b(String data){
		String rdata = "";
		//主
		String mainc = data.substring(0,1);

		int index1 = data.indexOf("(") +1;
		int index2 = data.indexOf(")");
		//副
		String fc = data.substring(index1,index2);

		int index3 = index2+3;
		String rc = data.substring(index3,index3+2);

		//结果档位
		String rn = rc.substring(1);;
		//结果
		rc = rc.substring(0,1);

		int index4 = index2 + 7;
		String beishu = "";
		if(index4 < data.length()){
			beishu = data.substring(index4, index4 + 1);
		}


		switch(mainc){
			case "B": rdata += B_str; break;
			case "b": rdata += b_str; break;
			case "j": rdata += j_str; break;
			case "c": rdata += c_str; break;
			case "G": rdata += G_str; break;
			case "g": rdata += g_str; break;
			case "S": rdata += S_str; break;
			case "s": rdata += s_str; break;
			case "F": rdata += F_str; break;
			case "y": rdata += y_str; break;
		}


		int fn = fc.length();
		for(int i=0;i<4;i++){
			String a;
			if(i<fn)
				a = fc.charAt(i)+"";
			else
				a = "";
			switch(a){
				case "B": rdata += B_str; break;
				case "b": rdata += b_str; break;
				case "j": rdata += j_str; break;
				case "c": rdata += c_str; break;
				case "G": rdata += G_str; break;
				case "g": rdata += g_str; break;
				case "S": rdata += S_str; break;
				case "s": rdata += s_str; break;
				case "F": rdata += F_str; break;
				case "f": rdata += f_str; break;
				default:  rdata += y_str; break;
			}
		}


		switch(rc){
			case "B": rdata += B_str; break;
			case "b": rdata += b_str; break;
			case "j": rdata += j_str; break;
			case "c": rdata += c_str; break;
			case "G": rdata += G_str; break;
			case "g": rdata += g_str; break;
			case "S": rdata += S_str; break;
			case "s": rdata += s_str; break;
			case "F": rdata += F_str; break;
			case "f": rdata += f_str; break;
			default: break;
		}

		switch(rn){
			case "1": rdata += n1_str; break;
			case "2": rdata += n2_str; break;
			case "3": rdata += n3_str; break;
			case "4": rdata += n4_str; break;
			default: break;
		}

		switch (beishu){
			case "2": rdata += b2_str; break;
			case "5": rdata += b5_str; break;
			default: rdata += b0_str; break;

		}
		return rdata;
	}

	//把一次强化记录转换为一个64位的数组输入样例：g(sjBb) b1
	public static float[] u1_float(String data){
		float[] rdata = y;
		//主
		String mainc = data.substring(0,1);

		int index1 = data.indexOf("(") +1;
		int index2 = data.indexOf(")");
		//副
		String fc = data.substring(index1,index2);

		int index3 = index2+3;
		String rc = data.substring(index3,index3+2);

		//结果档位
		String rn = rc.substring(1);
		//结果
		rc = rc.substring(0,1);


		switch(mainc){
			case "B": rdata = B; break;
			case "b": rdata = b; break;
			case "j": rdata = j; break;
			case "c": rdata = c; break;
			case "G": rdata = G; break;
			case "g": rdata = g; break;
			case "S": rdata = S; break;
			case "s": rdata = s; break;
			case "F": rdata = F; break;
			case "y": rdata = y; break;
			case "f": return null;
		}

		int fn = fc.length();
		for(int i=0;i<4;i++){
			String a;
			if(i<fn)
				a = fc.charAt(i)+"";
			else
				a = "";
			switch(a){
				case "B": rdata = linkArray(rdata,B); break;
				case "b": rdata = linkArray(rdata,b); break;
				case "j": rdata = linkArray(rdata,j); break;
				case "c": rdata = linkArray(rdata,c); break;
				case "G": rdata = linkArray(rdata,G); break;
				case "g": rdata = linkArray(rdata,g); break;
				case "S": rdata = linkArray(rdata,S); break;
				case "s": rdata = linkArray(rdata,s); break;
				case "F": rdata = linkArray(rdata,F); break;
				case "f": rdata = linkArray(rdata,f); break;
				default:  rdata = linkArray(rdata,y); break;
			}
		}

		switch(rc){
			case "B": rdata = linkArray(rdata,B); break;
			case "b": rdata = linkArray(rdata,b); break;
			case "j": rdata = linkArray(rdata,j); break;
			case "c": rdata = linkArray(rdata,c); break;
			case "G": rdata = linkArray(rdata,G); break;
			case "g": rdata = linkArray(rdata,g); break;
			case "S": rdata = linkArray(rdata,S); break;
			case "s": rdata = linkArray(rdata,s); break;
			case "F": rdata = linkArray(rdata,F); break;
			case "f": rdata = linkArray(rdata,f); break;
			default: break;
		}

		switch(rn){
			case "1": rdata = linkArray(rdata,n1); break;
			case "2": rdata = linkArray(rdata,n2); break;
			case "3": rdata = linkArray(rdata,n3); break;
			case "4": rdata = linkArray(rdata,n4); break;
			default: break;
		}
		return rdata;
	}

	//把一次强化记录转换为一个66位的数组输入样例,带倍数：g(sjBb) b1 1
	public static float[] u1_float__b(String data){
		float[] rdata = y;
		//主
		String mainc = data.substring(0,1);

		int index1 = data.indexOf("(") +1;
		int index2 = data.indexOf(")");
		//副
		String fc = data.substring(index1,index2);

		int index3 = index2+3;
		String rc = data.substring(index3,index3+2);

		//结果档位
		String rn = rc.substring(1);
		//结果
		rc = rc.substring(0,1);

		int index4 = index2 + 7;
		String beishu = data.substring(index4, index4 + 1);

		switch(mainc){
			case "B": rdata = B; break;
			case "b": rdata = b; break;
			case "j": rdata = j; break;
			case "c": rdata = c; break;
			case "G": rdata = G; break;
			case "g": rdata = g; break;
			case "S": rdata = S; break;
			case "s": rdata = s; break;
			case "F": rdata = F; break;
			case "y": rdata = y; break;
			case "f": return null;
		}

		int fn = fc.length();
		for(int i=0;i<4;i++){
			String a;
			if(i<fn)
				a = fc.charAt(i)+"";
			else
				a = "";
			switch(a){
				case "B": rdata = linkArray(rdata,B); break;
				case "b": rdata = linkArray(rdata,b); break;
				case "j": rdata = linkArray(rdata,j); break;
				case "c": rdata = linkArray(rdata,c); break;
				case "G": rdata = linkArray(rdata,G); break;
				case "g": rdata = linkArray(rdata,g); break;
				case "S": rdata = linkArray(rdata,S); break;
				case "s": rdata = linkArray(rdata,s); break;
				case "F": rdata = linkArray(rdata,F); break;
				case "f": rdata = linkArray(rdata,f); break;
				default:  rdata = linkArray(rdata,y); break;
			}
		}

		switch(rc){
			case "B": rdata = linkArray(rdata,B); break;
			case "b": rdata = linkArray(rdata,b); break;
			case "j": rdata = linkArray(rdata,j); break;
			case "c": rdata = linkArray(rdata,c); break;
			case "G": rdata = linkArray(rdata,G); break;
			case "g": rdata = linkArray(rdata,g); break;
			case "S": rdata = linkArray(rdata,S); break;
			case "s": rdata = linkArray(rdata,s); break;
			case "F": rdata = linkArray(rdata,F); break;
			case "f": rdata = linkArray(rdata,f); break;
			default: break;
		}

		switch(rn){
			case "1": rdata = linkArray(rdata,n1); break;
			case "2": rdata = linkArray(rdata,n2); break;
			case "3": rdata = linkArray(rdata,n3); break;
			case "4": rdata = linkArray(rdata,n4); break;
			default: break;
		}

		switch (beishu){
			case "2": rdata = linkArray(rdata,b2); break;
			case "5": rdata = linkArray(rdata,b5); break;
			default: rdata = linkArray(rdata, b0); break;

		}
		return rdata;
	}


	//把强化目标转位50位数组
	public static float[] t_float(String data){
		float[] rdata = y;
		//主
		String mainc = data.substring(0,1);

		int index1 = data.indexOf("(") +1;
		int index2 = data.indexOf(")");
		//副
		String fc = data.substring(index1,index2);
		switch(mainc){
			case "B": rdata = B; break;
			case "b": rdata = b; break;
			case "j": rdata = j; break;
			case "c": rdata = c; break;
			case "G": rdata = G; break;
			case "g": rdata = g; break;
			case "S": rdata = S; break;
			case "s": rdata = s; break;
			case "F": rdata = F; break;
			case "y": rdata = y; break;
			case "f": return null;
		}

		int fn = fc.length();
		for(int i=0;i<4;i++){
			String a;
			if(i<fn)
				a = fc.charAt(i)+"";
			else
				a = "";
			switch(a){
				case "B": rdata = linkArray(rdata,B); break;
				case "b": rdata = linkArray(rdata,b); break;
				case "j": rdata = linkArray(rdata,j); break;
				case "c": rdata = linkArray(rdata,c); break;
				case "G": rdata = linkArray(rdata,G); break;
				case "g": rdata = linkArray(rdata,g); break;
				case "S": rdata = linkArray(rdata,S); break;
				case "s": rdata = linkArray(rdata,s); break;
				case "F": rdata = linkArray(rdata,F); break;
				case "f": rdata = linkArray(rdata,f); break;
				default:  rdata = linkArray(rdata,y); break;
			}
		}
        return  rdata;
	}

	/*u2:把数据翻译成可输入编码
	  dataFolder:数据所在的文件夹
	  outFolder:输出的文件夹
	  data_len:每个训练数据有多少个(强化)记录
	  file_len:每个训练集文件放多少个训练数据*/
	public static void u2(String dataFolder,String outFolder,int data_len,int file_len){
		int alen = (data_len+1)*64;
		File dataPath = new File(dataFolder);
		File outPath = new File(outFolder);
		if( dataPath.exists() && dataPath.isFile()) return;
		File[] dataFile_list = dataPath.listFiles();
		if(dataFile_list.length==0) return;

		if(!outPath.exists() && !outPath.isDirectory())
			outPath.mkdir();//不存在就创建

		int file_n = 0;//第几个文件了
		int tfl = 0;//当前文件放了多少行
		File first_f = new File(outPath+"/0.txt");
		try{
			first_f.createNewFile();
			for(int i=0;i<dataFile_list.length;i++){
				ArrayList<String> dl = new ArrayList<String>();//未知数组长度，用list
				File file = dataFile_list[i];
				//System.out.println(Arrays.toString(dataFile_list));
				//File_tool.Readtxt(file.toString());
				if(file.isFile()){
					FileReader fileReader = new FileReader(file);
					BufferedReader in = new BufferedReader(fileReader);
					String line = null;
					int lin = 1;
					while ((line = in.readLine()) != null) {
						//System.out.println("this line：" + line);
						if(line!=null&&line.length()>3){
							String d =  u1(line);
							if(d.length() != 64 ) System.out.println("Loading data Eorro in " + file.toString() + " in line: " + lin + "   " + line );
							dl.add(d);//每个数据加入list
						}
						//line = null;
						lin++;
					}
					in.close();
					fileReader.close();
				}

				for(int j=0;j<dl.size();j++){
					String date = ""; //一条训练数据
					if( j + data_len+1 > dl.size() ) break;//超
					for(int k = j; k < j+data_len + 1; k++){
						date += dl.get(k);
					}

					/*if(date.length() != alen) {
						System.out.println("error: in " + dataFile_list[i].toString());
						System.out.println(j + " : " + dl.get(j));
					}*/

					if(tfl<file_len){
						File_tool.save_wirter(date,outPath+"/"+file_n+".txt");
						//System.out.println(file_n + "   " +tfl);
						tfl++;
					}else{
						file_n++;
						tfl = 0;
						new File(outPath+"/"+file_n+".txt").createNewFile();
					}
					//System.out.println(date);
				}//end for j */

			}// end for i
		}catch(Exception e){}
	}//end u2
	public static void u2__b(String dataFolder,String outFolder,int data_len,int file_len){

		File dataPath = new File(dataFolder);
		File outPath = new File(outFolder);
		if( dataPath.exists() && dataPath.isFile()) return;
		File[] dataFile_list = dataPath.listFiles();
		if(dataFile_list.length==0) return;

		if(!outPath.exists() && !outPath.isDirectory())
			outPath.mkdir();//不存在就创建

		int file_n = 0;//第几个文件了
		int tfl = 0;//当前文件放了多少行
		File first_f = new File(outPath+"/0.txt");
		try{
			first_f.createNewFile();
			for(int i=0;i<dataFile_list.length;i++){
				ArrayList<String> dl = new ArrayList<String>();//未知数组长度，用list
				File file = dataFile_list[i];
				//System.out.println(Arrays.toString(dataFile_list));
				//File_tool.Readtxt(file.toString());
				if(file.isFile()){
					FileReader fileReader = new FileReader(file);
					BufferedReader in = new BufferedReader(fileReader);
					String line = null;
					int lin = 1;
					while ((line = in.readLine()) != null) {
						//System.out.println("this line：" + line);
						if(line!=null&&line.length()>3){
							String d =  u1__b(line);
							if(d.length() != 66 ) System.out.println("Loading data Eorro in " + file.toString() + " in line: " + lin + "   " + line );
							dl.add(d);//每个数据加入list
						}
						//line = null;
						lin++;
					}
					in.close();
					fileReader.close();
				}

				for(int j=0;j<dl.size();j++){
					String date = ""; //一条训练数据
					if( j + data_len+1 > dl.size() ) break;//超
					for(int k = j; k < j+data_len + 1; k++){
						date += dl.get(k);
					}

					/*if(date.length() != alen) {
						System.out.println("error: in " + dataFile_list[i].toString());
						System.out.println(j + " : " + dl.get(j));
					}*/

					if(tfl<file_len){
						File_tool.save_wirter(date,outPath+"/"+file_n+".txt");
						//System.out.println(file_n + "   " +tfl);
						tfl++;
					}else{
						file_n++;
						tfl = 0;
						new File(outPath+"/"+file_n+".txt").createNewFile();
					}
					//System.out.println(date);
				}//end for j */

			}// end for i
		}catch(Exception e){}
	}//end u2


	public static void u2_rnn__b(String dataFolder, String outFolder, int min_timestep, int max_timestep){
		int end_len = max_timestep * (66 + 50) + 16;

		File datapath = new File(dataFolder);
		if (!datapath.exists()) return;
		if (!datapath.isDirectory()) return;

		File outPath = new File(outFolder);
		if(outPath.isFile() || !outPath.exists()) return;

		File datafile1 = new File(outPath+"/0.txt");

		try {
			datafile1.createNewFile();
			File[] datalistfile = datapath.listFiles();

			for(File datafile: datalistfile){

					if(!datafile.isFile()) continue;

					FileReader fileReader = new FileReader(datafile);
					BufferedReader bufferedReader = new BufferedReader(fileReader);
					int n = 1;
					String line_last = null;
					String line = null;
					ArrayList<String> embedding_data = new ArrayList<>();

					while ((line = bufferedReader.readLine()) != null){
						//embedding_data.add(u1__b(line));
						if(line!=null&&line.length()>3){
							//System.out.println("Loading data in " + datafile.toString() + " in line: " + n + "   " + line );
							String d =  u1__b(line);
							if(d.length() != 66 ) System.out.println("Loading data Eorro in " + datafile.toString() + " in line: " + n + "   " + line );
							embedding_data.add(d);//每个数据加入list
						}
						n++;
					}


					if(embedding_data.size() > min_timestep){
							StringBuilder endData = new StringBuilder("");

							for(int i=0; i<embedding_data.size()-1; i++){

									 endData.append(embedding_data.get(i)) //66位本次强化
											.append(embedding_data.get(i+1).substring(0,50)); //下一次强化的胚子

									if(i>=min_timestep-1) {
										StringBuilder ed = new StringBuilder(endData.toString() + embedding_data.get(i + 1).substring(50));

										while (ed.length() < end_len){
											ed.insert(0, "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
										}
										while (ed.length() > end_len){
											ed = new StringBuilder(ed.substring(116));
										}

										//System.out.println(ed.length());
										save_wirter(ed.toString(), datafile1.toString());
									}

							}//end for
					}//end if

					bufferedReader.close();
					fileReader.close();
			}

		}catch (Exception e){
			e.printStackTrace();
		}
	}


	//从文件读入训练集，转化为数组len:文件中数据个数,u2:data_len
	public static float[][][] loadData(String path,int fiel_len,int data_len){
		float[][][] r = new float[2][][];
		int input_n = data_len*64 + 50;
		float[][] input = new float[fiel_len][input_n];
		float[][] t = new float[fiel_len][14];
		
		File file = new File(path);
		if(file.isFile())
			try{
				FileReader fileReader = null;
				fileReader = new FileReader(file);
				BufferedReader in = new BufferedReader(fileReader);
				String line = null;
				int ln=0;
				while ((line = in.readLine()) != null) {
					//System.out.println( line.length());
					if(line.length()>5){
						for(int i=0;i<line.length();i++){
							char c = line.charAt(i);
							//System.out.println(c);
							float aa;
							if(c=='1'){
								aa = 1.0f;
								//System.out.println("bb");
							}else
								aa = 0;
								
							if(i<input_n)
								input[ln][i] = aa;
							else{
								t[ln][i-input_n] = aa;
								//System.out.println(aa);
							}
						}
						ln++;
					}
					line = null;
				}
				in.close();
				fileReader.close();
			}catch(Exception e){}
		r[0] = input;
		r[1] = t;
		return r;
	}
	public static float[][][] loadData__b(String path,int fiel_len,int data_len){
		float[][][] r = new float[2][][];
		int input_n = data_len*66 + 50;
		float[][] input = new float[fiel_len][input_n];
		float[][] t = new float[fiel_len][16];

		File file = new File(path);
		if(file.isFile())
			try{
				FileReader fileReader = null;
				fileReader = new FileReader(file);
				BufferedReader in = new BufferedReader(fileReader);
				String line = null;
				int ln=0;
				while ((line = in.readLine()) != null) {
					//System.out.println( line.length());
					if(line.length()>5){
						for(int i=0;i<line.length();i++){
							char c = line.charAt(i);
							//System.out.println(c);
							float aa;
							if(c=='1'){
								aa = 1.0f;
								//System.out.println("bb");
							}else
								aa = 0;

							if(i<input_n)
								input[ln][i] = aa;
							else{
								t[ln][i-input_n] = aa;
								//System.out.println(aa);
							}
						}
						ln++;
					}
					line = null;
				}
				in.close();
				fileReader.close();
			}catch(Exception e){}
		r[0] = input;
		r[1] = t;
		return r;
	}
	
	
	//结果反编码
	public static String recode(String code){
		String rc = code.substring(0,10);
		String rn = code.substring(10);
		int index1=-1,index2=-1;
		for(int i=0;i<rc.length();i++)
			if(rc.charAt(i)=='1'){
				index1=i;
				break;
			}
		for(int i=0;i<rn.length();i++)
			if(rn.charAt(i)=='1'){
				index2=i;
				break;
			}
		switch(index1){
			case 0: rc = "B"; break;
			case 1: rc = "b"; break;
			case 2: rc = "j"; break;
			case 3: rc = "c"; break;
			case 4: rc = "G"; break;
			case 5: rc = "g"; break;
			case 6: rc = "S"; break;
			case 7: rc = "s"; break;
			case 8: rc = "F"; break;
			case 9: rc = "f"; break;
		}
		
		switch(index2){
			case 0: rc+="1"; break;
			case 1: rc += "2"; break;
			case 2: rc += "3"; break;
			case 3: rc += "4"; break;
		}
		return rc;
	}
	public static String recode(float[] code){//14
		String rc ="";
		int index1=-1,index2=-1;
		float max=0;
		for(int i=0;i<10;i++)
			if(code[i]>max){
				index1=i;
				max = code[i];
			}
		max = 0;
		for(int i=0;i+10<code.length;i++){
			if(code[i+10]>max){
				index2=i;
				max = code[i+10];
			}
			//System.out.println(code[i+10]);
		}
		switch(index1){
			case 0: rc = "B"; break;
			case 1: rc = "b"; break;
			case 2: rc = "j"; break;
			case 3: rc = "c"; break;
			case 4: rc = "G"; break;
			case 5: rc = "g"; break;
			case 6: rc = "S"; break;
			case 7: rc = "s"; break;
			case 8: rc = "F"; break;
			case 9: rc = "f"; break;
			default:rc = "y"; break;
		}

		switch(index2){
			case 0: rc+="1"; break;
			case 1: rc += "2"; break;
			case 2: rc += "3"; break;
			case 3: rc += "4"; break;
			default:rc += ""; break;
		}
		return rc;
	}


	//数组连接
	public static float[] linkArray(float[] array1, float[] array2) {
		if (array1 == null) {
			return array2;
		}
		if (array2 == null) {
			return array1;
		}
		float[] list = new float[array1.length+array2.length];
		for (int i = 0; i < array1.length; i++) {
			list[i] = array1[i];
		}
		int j=0;
		for (int i = array1.length; i < list.length; i++) {
			list[i] = array2[j];
			j++;
		}
		return list;
	}




	/**
	 * 数据增强
	 * @param souricesPath 数据源目录
	 * @param outPath 输出目录
	 */
	public static void updata(String souricesPath, String outPath, int n){

		File sourceFilePath = new File(souricesPath);
		if (!sourceFilePath.exists()) return;
		if(!sourceFilePath.isDirectory()) return;

		File[] datalistFile = sourceFilePath.listFiles();
		if( datalistFile.length == 0) return;

		File outPathFile = new File(outPath);

		if (!outPathFile.exists() || outPathFile.isFile()) outPathFile.mkdirs();

		ArrayList[] updatalist = new ArrayList[n];

		for(File dataFile: datalistFile){
			System.out.println(dataFile.toString());
			for(int i=0; i < updatalist.length; i++)
				updatalist[i] = new ArrayList<String>();

			try {
				FileReader fileReader = new FileReader(dataFile);
				BufferedReader in = new BufferedReader(fileReader);

				String line = null;
				while ((line = in.readLine()) != null){
					char[] data = line.toCharArray();

					//System.out.println(data.length);
					if (data.length == 0) continue;

					int fcn = 0;
					int index = 2;
					if(data.length < 3) {
						System.out.println("Error in " + dataFile + " in line: " + line );
					}
					while(data[index + fcn] != ')') fcn++;

					char[] fc = new char[fcn];
					for(int i=0; i < fcn; i++) fc[i] = data[i+index];

					//System.out.println(Arrays.toString(fc) + "  " + line);


					if (fcn <= 1){
						for (ArrayList arrayList : updatalist) arrayList.add(line);
					}else{
						for (ArrayList arrayList : updatalist) {
							char[] newfc = reSort(fc);
							System.arraycopy(newfc, 0, data, 0 + index, fcn);
							//System.out.println(data.toString());
							arrayList.add(String.valueOf(data));
						}
					}

				}



		     	//File[] outfile = new File[n];
			   for (int i=0; i < n; i++){
			   	  String filename = outPath + "\\" + dataFile.toString().substring(dataFile.toString().lastIndexOf("\\")+1,dataFile.toString().lastIndexOf(".")) + "___" + i +".txt";
				  //System.out.println(dataFile.toString().substring(dataFile.toString().lastIndexOf("\\")+1,dataFile.toString().lastIndexOf(".")));
				  File outfile = new File(filename);
				  outfile.createNewFile();
				  for(Object data: updatalist[i]) save_wirter(data.toString(), filename);
			   }

			}catch (IOException ignored){}
		}

	}

	private static void save_wirter(String s, String path) {
		//向文件追加内容
		//System.out.println(s);
		File f=new File(path);
		if(f.isFile()){
			FileWriter fw = null;
			try {
				fw = new FileWriter(f, true);
			}catch (IOException e) {
				e.printStackTrace();
			}
			PrintWriter pw = new PrintWriter(fw);
			pw.println(s);
			pw.flush();
			try {
				fw.flush();
				pw.close();
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.out.println(e);
			}
		}}


	/**
	 * 随机打乱数组
	 * @param originalData s
	 * @return  s
	 */
	public static int[] reSort(int[] originalData){
		int len = originalData.length;
		//使用日期做种子生成真随机数
		Random random = new Random();
		int index = Math.abs(random.nextInt() % len);//限制到0-len
		//使用HashSet来保证index的唯一性
		HashSet<Integer> set = new HashSet<>();
		//新数组存储打乱后的元素
		int[] newNums = new int[len];
		int j = 0;
		for (int i=0; i < len; i ++){
			//一直寻找新的index如果index已经存在
			while(!set.add(index)){
				//index存在时add方法为false
				index = Math.abs(random.nextInt() % len);
			}
			//将index位置的元素放进新数组
			newNums[j++] = originalData[index];
		}
		return newNums;
	}
	public static char[] reSort(char[] originalData){
		int len = originalData.length;
		//使用日期做种子生成真随机数
		Random random = new Random();
		int index = Math.abs(random.nextInt() % len);//限制到0-len
		//使用HashSet来保证index的唯一性
		HashSet<Integer> set = new HashSet<>();
		//新数组存储打乱后的元素
		char[] newNums = new char[len];
		int j = 0;
		for (int i=0; i < len; i ++){
			//一直寻找新的index如果index已经存在
			while(!set.add(index)){
				//index存在时add方法为false
				index = Math.abs(random.nextInt() % len);
			}
			//将index位置的元素放进新数组
			newNums[j++] = originalData[index];
		}
		return newNums;
	}


	public static void main(String[] args) {

//		String souricesPath = "F:\\java Project\\圣遗物强化分析\\圣遗物强化数据\\历史";
//		String outPath = "F:\\java Project\\圣遗物强化分析\\圣遗物强化数据\\增强数据";
//		updata(souricesPath,outPath,3);
//

		String souriceDataPath = "F:\\java Project\\圣遗物强化分析\\圣遗物强化数据\\历史";
		String op10Path = "F:\\java Project\\圣遗物强化分析\\圣遗物强化数据\\强化记录data_len=12";

//		u2_rnn__b(souriceDataPath,op10Path,10,30);
//
//		File df = new File(op10Path+"\\0.txt");
//		if(df.exists()) {
//			boolean delete = df.delete();
//		}
//

		u2(souriceDataPath,op10Path,12,999999);


	}

}
