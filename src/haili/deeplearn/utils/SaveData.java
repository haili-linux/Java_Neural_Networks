package haili.deeplearn.utils;

import java.util.Arrays;

public class SaveData {
    public static String sInt(String name, int value){
        return name+":"+value;
    }
    public static int getSInt(String value){
        return Integer.parseInt( value.substring(value.indexOf(":")+1));
    }
    public static String sFloat(String name, float value){
        return name+":"+value;
    }
    public static float getSFloat(String value){
        return Float.parseFloat(value.substring(value.indexOf(":")+1));
    }
    public static String sIntArrays(String name, int[] data){
        return name + ":length:" + data.length + " " + Arrays.toString(data);
    }
    public static String sFloatArrays(String name,float[] data){
        return name + ":length:" + data.length + " " + Arrays.toString(data);
    }
    public static int[] getsIntArrays(String s){
        int length = Integer.parseInt( s.substring(s.lastIndexOf(":")+1,s.indexOf("[")-1));
        int[] sd=new int[length];
        int n=0;
        char d = ',';
        for(int i=0;i<s.length();i++){
            if(s.charAt(i)=='[' || s.charAt(i)==','){
                i++;
                StringBuilder data= new StringBuilder();
                while(s.charAt(i)!=d && s.charAt(i)!=']'){
                    data.append(s.charAt(i));
                    i++;
                }
                sd[n] =(int) Double.parseDouble(data.toString());
                n++;
                i--;
            }
        }
        return sd;
    }
    public static float[] getsFloatArrays(String s){
        int i1 = s.lastIndexOf("[");
        int length = Integer.parseInt( s.substring(s.lastIndexOf(":")+1,i1-1));

        s = s.substring(i1 + 1, s.length()-1);


        float[] data = new float[length];
        for (int i = 0; i < data.length; i++){
            int index =  s.indexOf(',');

            if(index != -1) {
                data[i] = Float.parseFloat(s.substring(0, index));
                s = s.substring(index+1);
            } else {
                data[i] = Float.parseFloat(s);
                break;
            }
        }

        return data;
    }

    public static void main(String[] args) {


    }
}
