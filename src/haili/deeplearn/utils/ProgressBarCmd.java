package haili.deeplearn.utils;

import java.util.Arrays;
import java.util.Collections;

public class ProgressBarCmd {

    int pixel_number;
    int total;

    String title;
    String backString;
    int lastLen = 0;

    public ProgressBarCmd(int total, int pixel_number){
        title = "";
        this.total = total;

        //最小10个单位
        if(pixel_number < 10) pixel_number = 10;
        this.pixel_number = pixel_number;

        char[] chars = new char[pixel_number + 2 + 1 + 3 + 1];
        Arrays.fill(chars, '\b');
        backString = new String(chars);
    }

    public ProgressBarCmd(String title, int total, int pixel_number){
        this.title = title;
        this.total = total;

        //最小10个单位
        if(pixel_number < 10) pixel_number = 10;
        this.pixel_number = pixel_number;

        char[] chars = new char[title.length() + pixel_number + 2 + 1 + 3 + 1];
        Arrays.fill(chars, '\b');
        backString = new String(chars);

    }

    public String setProgress(int died_number){
        //[=_________]
        double percentage = (double) died_number / total;
        int p = (int) (percentage * pixel_number);

        char[] did = new char[p];
        Arrays.fill(did, '=');
        if( p > 0 &&  p != pixel_number) did[p-1] = '>';

        char[] wait = new char[pixel_number - p];
        Arrays.fill(wait, '.');

        double percentage_double = percentage * 100;
        int percentage_int = (int) percentage_double;

        String p_str = "" + percentage_int;
        if(p_str.length() > 5)
            p_str = p_str.substring(0, 5);

        if(percentage_int < 10) p_str = "  " + p_str;
        else if(percentage_int < 100) p_str = " " + p_str;



        String progress = title + "[" + new String(did) + new String(wait) + "] " + p_str + "%";
        //lastLen = progress.length();



        return backString + progress;
    }

//    public static void main(String[] args) throws Exception{
//        //simple
//        ProgressBarCmd progressBarCmd = new ProgressBarCmd("simple: ",100, 25);
//        System.out.println("progressbar: ");
//        for (int  i = 1; i <= 100; i++){
//            System.out.print(progressBarCmd.setProgress(i));
//            Thread.sleep(100);
//        }
//    }
}
