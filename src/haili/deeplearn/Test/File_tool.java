package haili.deeplearn.Test;

import java.io.*;

public class File_tool
{
	public static void save_wirter(String s,String path) {
		//向文件追加内容
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
		  }
	  }}
	  
	  
    public static void Readtxt(String path){
		  //String path= "F://Forward.txt";
		  File file = new File(path);
		  if(file.isFile())
		  try{
		  FileReader fileReader = null;
		  fileReader = new FileReader(file);
		  BufferedReader in = new BufferedReader(fileReader);
		  String line = null;
		  while ((line = in.readLine()) != null) {
			  System.out.println("this line：" + line);
			  line = null;
		  }
		  in.close();
		  fileReader.close();
          }catch(Exception e){}
	  }
	  
	 //读取一行
	public static String Readtxt(String path,int n){
		//String path= "F://Forward.txt";
		String r=null;
		File file = new File(path);
		if(file.isFile())
			try{
				FileReader fileReader = null;
				fileReader = new FileReader(file);
				BufferedReader in = new BufferedReader(fileReader);
				String line = null;
				int l=1;
				while ((line = in.readLine()) != null) {
					   
					   if(l==n){
						   r=line;
						   break;
						}
					   l++;
				}
				in.close();
				fileReader.close();
			}catch(Exception e){}
		return r;
	}
	  
	public static int count(String filename) throws IOException {
		InputStream is = new BufferedInputStream(new FileInputStream(filename));

		try {
			byte[] c = new byte[1024];

			int count = 0;

			int readChars = 0;

			while ((readChars = is.read(c)) != -1) {
				for (int i = 0; i < readChars; ++i) {
					if (c[i] == '\n')

						++count;

				}

			}

			return count;

		} finally {
			is.close();

		}

	}
}
