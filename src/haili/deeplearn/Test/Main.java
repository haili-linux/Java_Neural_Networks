package haili.deeplearn.Test;

import haili.deeplearn.BpNetwork;
import haili.deeplearn.DeltaOptimizer.Adam;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.*;

public class Main {

    public static void main(String[] args){
        int datalen = 12;
        String souriceDataPath = "F:\\java Project\\原神圣遗物强化分析\\圣遗物强化分析\\圣遗物强化数据";
        String op10Path = "F:\\java Project\\原神圣遗物强化分析\\圣遗物强化分析\\圣遗物强化数据\\强化记录data_len=12";
        //String op10Path = "F:\\java Project\\圣遗物强化分析\\圣遗物强化数据\\t\\d";
        //LoadData.u2(souriceDataPath,op10Path,datalen,8000);
        String savePath = "F:\\java Project\\原神圣遗物强化分析\\art分析3\\保存的\\";
        //String souriceDataPath = "圣遗物强化数据";
        //String op10Path = "强化记录data_len=12";
        //String savePath = "保存的\\";




        File dataFile = new File(op10Path);
        File[] datalist = dataFile.listFiles();
        float[][][][] data = new float[datalist.length][][][];


        try {
            for (int j = 0; j < datalist.length; j++)
                data[j] = LoadData.loadData(datalist[j].toString(),File_tool.count(datalist[j].toString()), datalen);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(data[0][0].length);

        System.out.println();

//
//        int[] arf = new int[]{818, 512, 64, 64, 32};
//        BpNetwork bpNetwork = new BpNetwork(818, 14, 0.02f, new LRelu(), arf);
//
//        for (int i = 0; i < bpNetwork.output_Neuer.length; i++)
//            bpNetwork.output_Neuer[i].ACT_function = new Sigmoid();


        ///BpNetwork bpNetwork = new BpNetwork(savePath + "1.txt");
        BpNetwork bpNetwork = new BpNetwork("F:\\java Project\\tf_py_demo\\test2.txt");

        bpNetwork.lr = 0.000001f;
        bpNetwork.deltaOptimizer = new Adam(bpNetwork.getWandD_number(), 0.9f,0.999f,1e-8f);
        Scanner sc = new Scanner(System.in);
        while (true) {
            try {
                System.out.print("Enter Command: ");
                String command = sc.next();
                switch (command) {
                    case "load" :
                        File df = new File(op10Path+"\\0.txt");
                        if(df.exists()) df.delete();

                        LoadData.u2(souriceDataPath,op10Path,datalen,99999999);
                        dataFile = new File(op10Path);
                        datalist = dataFile.listFiles();
                        data = new float[datalist.length][][][];

                        int l = 0;
                        try {
                            for (int j = 0; j < datalist.length; j++)
                                data[j] = LoadData.loadData(datalist[j].toString(),l = File_tool.count(datalist[j].toString()), datalen);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        System.out.println("len: " + l);

                        break;

                    case "error":
                        System.out.println(bpNetwork.calculateLoss(data[0][0], data[0][1]));
                        break;

                    case "sleep":
                        System.out.print("输入睡眠时间min: ");
                        int time = sc.nextInt();
                        try {
                            Thread.sleep(time * 1000 * 60 );
                        }catch (Exception e){}
                        System.out.println("start");
                        break;

                    case "save"://保存
                        String path = savePath + "1.txt";
                        File file = new File(path);
                        if(file.exists()) {
                            file.delete();
                        }
                        bpNetwork.saveInFile(path);
                        System.out.println("已保存");
                        break;

                    case "learnt":
                        System.out.print("enter: Time(min), batch-size, thread ");
                        int time2, batcht, tnt;
                        time2 = sc.nextInt();
                        batcht = sc.nextInt();
                        tnt = sc.nextInt();
                        System.out.println("  " + getTime() );
                        bpNetwork.fit_time(data[0][0], data[0][1], batcht, tnt,time2);
                        System.out.println("  loss:" + bpNetwork.dError);
                        System.out.println("  " + getTime() );
                        break;

                    case "learn":  //学习
                        System.out.print("enter: epoch, batch-size, thread ");
                        int n, batch, tn;
                        n = sc.nextInt();
                        batch = sc.nextInt();
                        tn = sc.nextInt();

                        if(batch < 10) {
                            System.out.print("batch小于过小,是否继续Y/N? ");
                            if(!sc.next().equals("Y")) break;
                        }
                        long t1 = System.currentTimeMillis();

                        bpNetwork.fit(data[0][0], data[0][1], batch, n, tn);
                        System.out.println("   " + bpNetwork.dError);

                        t1 = System.currentTimeMillis() - t1;

                        System.out.println("   time:" + t1 );
                        Date d = new Date();
                        SimpleDateFormat sbf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                        System.out.println(sbf.format(d));
                        break;



                    case "out":
                        ArrayList<float[]> arrayList = new ArrayList<float[]>();
                        ArrayList<String> strList = new ArrayList<String>();
                        while (true) {
                            if (arrayList.size() < 12) {
                                System.out.print("输入强化结果" + (arrayList.size() + 1) + ": ");
                                String a = sc.next();

                                strList.add(a.substring(0,a.lastIndexOf(")")+1) +"  "+ a.substring(a.lastIndexOf(")")+3) + " ×1  " + Main.getTime());

                                arrayList.add(LoadData.u1_float(a));
                            } else {
                                float[] in = arrayList.get(0);
                                for (int i = 1; i < arrayList.size(); i++)
                                    in = LoadData.linkArray(in, arrayList.get(i));

                                while (true) {
                                    System.out.print("输入强化目标或输入0退出:");
                                    String ins = sc.next();
                                    if(ins.equals("0")) break;
                                    float[] ind = LoadData.linkArray(in, LoadData.t_float(ins));
                                    float[] out = bpNetwork.out_(ind);
                                    for (int i = 0; i < out.length; i++) {
                                        if (i < LoadData.ct.length())
                                            System.out.println(LoadData.ct.charAt(i) + ":" + out[i]);
                                        else System.out.println(i - 9 + ":" + out[i]);
                                    }
                                    System.out.println("预测结果: " + LoadData.recode(out));
                                    System.out.println("__________________________________");
                                }

                                System.out.print("输出0保存并退出或实际结果:");
                                String a = sc.next();

                                if(a.equals("0")){
                                    saData(strList);
                                    break;
                                }

                                strList.add(a.substring(0,a.lastIndexOf(")")+1) +"  "+ a.substring(a.lastIndexOf(")")+3) + " ×1  " + Main.getTime());
                                float[] t = LoadData.u1_float(a);

                                arrayList.remove(0);
                                arrayList.add(t);
                            }
                        }

                    case "test":
                        for (int j = 0; j<10;j++){
                            int index = (int)(Math.random()*data[0][0].length);
                            float[] mc = new float[10];
                            float[] f1 = new float[10];
                            float[] f2 = new float[10];
                            float[] f3 = new float[10];
                            float[] f4 = new float[10];
                            System.arraycopy(data[0][0][index],bpNetwork.input_n-50,mc,0,10);
                            System.arraycopy(data[0][0][index],bpNetwork.input_n-40,f1,0,10);
                            System.arraycopy(data[0][0][index],bpNetwork.input_n-30,f2,0,10);
                            System.arraycopy(data[0][0][index],bpNetwork.input_n-20,f3,0,10);
                            System.arraycopy(data[0][0][index],bpNetwork.input_n-10,f4,0,10);

                            String m = LoadData.recode(mc);
                            String f1s = LoadData.recode(f1);
                            String f2s = LoadData.recode(f2);
                            String f3s = LoadData.recode(f3);
                            String f4s = LoadData.recode(f4);
                            String a = m + f1s + f2s + f3s + f4s;
                            //System.out.println(data[0][0][index].length + Arrays.toString(data[0][0][index]));
                            float[] out = bpNetwork.out_(data[0][0][index]);
                            float[] out2 = out.clone();

                            String target = m + "(" ;
                            if(!f1s.equals("y")) target = target + f1s;
                            if(!f2s.equals("y")) target = target + f2s;
                            if(!f3s.equals("y")) target = target + f3s;
                            if(!f4s.equals("y")) target = target + f4s;
                            target = target + ")";

                            String fc =  f1s + f2s + f3s + f4s;
                            System.out.println("r: "  + target);

                            String chati = "";
                            if (target.length()<7){
                                for (int i = 0; i < LoadData.ct.length(); i++){
                                    chati = LoadData.ct.charAt(i)+"";
                                    if(fc.contains(chati)) out[i] = 0;
                                    if(m.contains(chati)) out[i] = 0;
                                }
                            }else{
                                for (int i = 0; i < LoadData.ct.length(); i++){
                                    chati = LoadData.ct.charAt(i)+"";
                                    if(!fc.contains(chati)) out[i] = 0;
                                    if(m.contains(chati)) out[i] = 0;
                                }
                            }

                            System.out.println(out_toString(out));
                            System.out.println(out_toString2(out2));


//                            String r1="",r2="";
//                            for (int i = 0; i < out.length; i++) {
//                                if (i < LoadData.ct.length()) {
//                                    String c = LoadData.ct.charAt(i)+"";
//                                    int len = 0;
//                                    //String tc =
//                                    for(int k=1;k<a.length();k++)
//                                        if(!(a.charAt(k)+"").equals("y"))
//                                            len++;
//
//                                    if(len<4) {
//                                        if (c.equals(m) || c.equals(f1s) || c.equals(f2s) || c.equals(f3s) || c.equals(f4s))
//                                            ;
//                                        else
//                                            r1 += LoadData.ct.charAt(i) + ":" + (out[i] + "").substring(0, 5) + "  ";
//                                    }else{
//                                        if( c.equals(f1s) || c.equals(f2s) || c.equals(f3s) || c.equals(f4s))
//                                            r1 += LoadData.ct.charAt(i) + ":" + (out[i] + "").substring(0, 5) + "  ";
//                                    }
//                                }else r2 += (i - 9) + ":" + (out[i]+"").substring(0,5)+"  ";
//                            }
//                            System.out.println(r1);
//                            System.out.println("sssssssss" +r2);
                            System.out.println(index +"  " +a+"   真实值:"+ LoadData.recode(data[0][1][index]) +"   预测值:"+ LoadData.recode(out));
                            System.out.println("");
                        }
                        break;
                }

            }catch (Exception e){
                e.printStackTrace();
                System.out.println(e);
            }
            //continue;
        }//end while */

    }


    private static String out_toString(float[] out){
        String r = "";
        float tt = 0;
        float dd = 0;
        for (int i = 0; i < 10; i++) tt += out[i];
        for (int i = 10; i < out.length; i++)  dd += out[i];

        for (int i = 0; i < 10; i++)  out[i] /= tt;
        for (int i = 10; i < out.length; i++) out[i] /= dd;


        for (int i = 0; i < out.length; i++) {
            if (i < LoadData.ct.length())
                r += LoadData.ct.charAt(i) + ":" + d(out[i])+ "    ";
            else r +=  i - 9 + "档:" + d(out[i]) + "    ";
        }
        //  System.out.println("预测结果: " + LoadData.recode(out));
        return  r;
    }
    private static String out_toString2(float[] out){
        StringBuilder r = new StringBuilder("");

        for (int i = 0; i < out.length; i++) {
            if (i < LoadData.ct.length())
                r.append(LoadData.ct.charAt(i) + ":" + d(out[i])+ "    ");
            else r.append(i - 9 + "档:" + d(out[i]) + "    ");
        }
        //  System.out.println("预测结果: " + LoadData.recode(out));
        return  r.toString();
    }
    public static String d(float a){
        String r = "" + a;
        return r;
    }

    public static void saData( ArrayList<String> dataList){
        File path = new File("圣遗物强化记录");
        if (path.exists() && path.isDirectory());
        else path.mkdir();
        String filename = path + "\\强化记录_" + getTime() + ".txt";
        File jl = new File(filename);
        try {
            jl.createNewFile();
        }catch (IOException e){}
        for(String data: dataList) save_wirter(data,filename);
    }
    public static void save_wirter(String s,String path) {
        //向文件追加内容
        System.out.println(s);
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
    public static String getTime() {
        Calendar calendars = Calendar.getInstance();
        calendars.setTimeZone(TimeZone.getTimeZone("GMT+8:00"));
        String year = String.valueOf(calendars.get(Calendar.YEAR));
        String month = String.valueOf(1 + calendars.get(Calendar.MONTH));
        String day = String.valueOf(calendars.get(Calendar.DATE));
        String hour = String.valueOf(calendars.get(Calendar.HOUR));
        String min = String.valueOf(calendars.get(Calendar.MINUTE));
        String second = String.valueOf(calendars.get(Calendar.SECOND));
        //Boolean isAm = calendars.get(Calendar.AM_PM)==1 ? true:false;
        //Boolean is24 = DateFormat.is24HourFormat(getApplication()) ?true:false;
        return year+"年"+month+"月"+day+"日"+hour+"时"+min+"分"+second;
    }
}
