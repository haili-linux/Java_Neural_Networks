package haili.deeplearn.utils;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ThreadWork {

    private static ExecutorService executorService = null;//Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    public static synchronized void start(ThreadWorker threadWorker){

        if(executorService == null) executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        //System.out.println("start");

        Future[] futureList = new Future[Runtime.getRuntime().availableProcessors()];

        for(int i=0; i < futureList.length; i++) {
            futureList[i] = executorService.submit(new Thread(threadWorker));
            //System.out.println("start" + i);
        }

        try {
            for(Future future: futureList) future.get();
        }catch (InterruptedException | ExecutionException e){
            e.printStackTrace();
            System.exit(0);
        }
    }

    public static synchronized void start(ThreadWorker threadWorker, int threadNumber){

        if(executorService == null) executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        //System.out.println("start");

        Future[] futureList = new Future[threadNumber];
       // System.out.println("         (" + futureList.length);

        for(int i = 0; i < threadNumber; i++) {
            futureList[i] = executorService.submit(new Thread(threadWorker));
            //System.out.println("start" + i);
        }

        try {
            for(Future future: futureList) future.get();

        }catch (InterruptedException | ExecutionException e){
            e.printStackTrace();
        }
    }

    //批量梯度下降，多线程并行计算
    public static class ThreadWorker implements Runnable{
        private boolean finish = false;
        private final boolean[] finishNumber;//是否计算完成
        private final boolean[] lock;//是否锁了(正在计算)

        public ThreadWorker(int workNumber){
            this.finishNumber = new boolean[workNumber];
            this.lock = new boolean[workNumber];
        }

        public final int getWorkNumber(){ return finishNumber.length; }

        int start_index = 0;

        public final void run(){
            while(true){
                if(finish) break;
                boolean flag = true;
                for(int p = start_index; p < finishNumber.length; p++)
                    if(!finishNumber[p]){	//第i个未完成
                        flag = false;
                        if(!lock[p]){//i个未锁
                            lock[p] = true; //上锁
                            //doing
                            working(p);
                            //处理完成,
                            finishNumber[p] = true;
                        }

                    }
                if(flag){
                    finish = true;//已完成
                    break;
                }
            }//end while
        }//end run


        public void setStart_index(int start_index) {
            this.start_index = start_index;
        }

        public void working(int index){
            //doing
        }

        public Object getObject(){
            return null;
        }

    }


    /*
    public static void main(String[] args) {
        ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(10){
            int[] ints = new int[getWorkNumber()];
            @Override
            public void working(int index) {
                try {
                    Thread.sleep(200);
                } catch (InterruptedException exception) {
                    exception.printStackTrace();
                }
                ints[index] = index;
            }

            @Override
            public Object getObject() {
                return ints;
            }
        };

        long t0 = System.currentTimeMillis();
        ThreadWork.start(threadWorker);
        t0 = System.currentTimeMillis() - t0;

        System.out.println("use time: " + t0);

        int[] d = (int[]) threadWorker.getObject();
        Arrays.sort(d);
        System.out.println(Arrays.toString(d));

        //
        for( int i = 0; i < d.length; i++){
            if( i!= d[i] ) System.out.println(i);
        }
        System.out.println("end");

    }

     */



}
