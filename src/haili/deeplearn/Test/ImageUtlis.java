package haili.deeplearn.Test;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

public class ImageUtlis {
    public static void main(String[] args) throws Exception {

         //autoInput.CutScreenShot.screenShot("data");

//        BufferedImage image = ImageIO.read(new File("data\\1.png"));
//        Image i1 = image.getScaledInstance(20, 20, Image.SCALE_DEFAULT);

        ArrayList<BufferedImage> bufferedImageArrayList = cutImage("data\\h500___0.8161053523475125.png",25, 25);
        String path = "data\\cutimg\\";
        int index = 1;
        for (BufferedImage bfimg: bufferedImageArrayList){
            ImageIO.write(bfimg,"png", new File(path + index + ".png"));
            index++;
        }
    }

    /**
     * bmp图片像素转数组
     * @param imgsrc bmp图片
     * @return 。
     * @throws Exception null
     */
    public static float[] bmpToRgbList(BufferedImage imgsrc) {
        float[] rgb = new float[imgsrc.getWidth() * imgsrc.getHeight()];

        //创建一个灰度模式的图片
        BufferedImage back = new BufferedImage(imgsrc.getWidth(), imgsrc.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        int width = imgsrc.getWidth();
        int height = imgsrc.getHeight();

        int index = 0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                back.setRGB(i, j, imgsrc.getRGB(i, j));
                rgb[index] = (back.getRGB(i, j) &  0xff) / 255.0f;
                index++;
            }
        }
        return rgb;
    }

    public static float[] bmpToRgbList(String filename) throws Exception {
        BufferedImage imgsrc = ImageIO.read(new File(filename));
        return  bmpToRgbList(imgsrc);
    }


    /**
     * 颜色取反
     */
    public static float[] bmpToRgbList_F(BufferedImage imgsrc) {
        float[] rgb = new float[imgsrc.getWidth() * imgsrc.getHeight()];
        //创建一个灰度模式的图片
        BufferedImage back = new BufferedImage(imgsrc.getWidth(), imgsrc.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        int width = imgsrc.getWidth();
        int height = imgsrc.getHeight();

        int index = 0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                back.setRGB(i, j, imgsrc.getRGB(i, j));
                rgb[index] =( 255.0f - (back.getRGB(i, j) &  0xff) ) / 255.0f;
                index++;
            }
        }
        return rgb;
    }



    //把图片分割成n个 w * h 的图片
    public static ArrayList<BufferedImage> cutImage(BufferedImage img, int w, int h) throws Exception {
        ArrayList<BufferedImage> imageArrayList = new ArrayList<>();
        int width = img.getWidth();
        int height = img.getHeight();

        for(int hi = 0; hi < height; hi += h){
            for (int wi = 0; wi < width; wi += w){
                int twi = wi, thi = hi;
                if(wi + w > width){
                    twi = width - w;
                }
                if(hi + h > height){
                    thi = height - h;
                }
                BufferedImage bufferedImage = img.getSubimage(twi, thi, w, h);
                imageArrayList.add(bufferedImage);
            }
        }

        return imageArrayList;
    }
    public static ArrayList<BufferedImage> cutImage(String filename, int w, int h) throws Exception {
        File imgfile = new File(filename);
        if (!imgfile.exists() || imgfile.isDirectory())  return null;
        BufferedImage img = ImageIO.read(imgfile);
        return cutImage(img, w, h);
    }


    //图片缩放
    public static BufferedImage ScaledInstance(BufferedImage bufferedImage, double bn){
        Image i1 = bufferedImage.getScaledInstance((int)(bufferedImage.getWidth() * bn), (int)(bufferedImage.getHeight() * bn), Image.SCALE_DEFAULT);
        return toBufferedImage(i1);
    }
    public static BufferedImage ScaledInstance(BufferedImage bufferedImage, int w, int h){
        Image i1 = bufferedImage.getScaledInstance(w, h, Image.SCALE_DEFAULT);
        return toBufferedImage(i1);
    }


    //转换为灰度图
    public static BufferedImage getGrayImage(BufferedImage b) {//定义灰度方法  返回值为BufferedImage对象
        int width = b.getWidth();
        int height =b.getHeight();
        BufferedImage bufferedImage_end = new BufferedImage(width,height, BufferedImage.TYPE_3BYTE_BGR );  //构建新的对象模型
        // 遍历图片的RGB值，把得到的灰度值存到bufferedImage_end中，然后返回bufferedImage_end
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(b.getRGB(x,y));//构建Color获取图片像素点
                int gray = (int)(color.getRed() * 0.299 + color.getGreen() * 0.587 + color.getBlue() *0.114);
                Color color_end = new Color(gray,gray,gray);   //将设置的像素设置到bufferedImage_end
                bufferedImage_end.setRGB(x,y,color_end.getRGB());
            }
        }
        return bufferedImage_end;
    }


    private static BufferedImage toBufferedImage(Image image) {
        if (image instanceof BufferedImage) {
            return (BufferedImage) image;
        }
        image = new ImageIcon(image).getImage();
        boolean hasAlpha = false;
        BufferedImage bimage = null;
        GraphicsEnvironment ge = GraphicsEnvironment
                .getLocalGraphicsEnvironment();
        try {
            int transparency = Transparency.OPAQUE;
            if (hasAlpha) {
                transparency = Transparency.BITMASK;
            }
            GraphicsDevice gs = ge.getDefaultScreenDevice();
            GraphicsConfiguration gc = gs.getDefaultConfiguration();
            bimage = gc.createCompatibleImage(image.getWidth(null), image
                    .getHeight(null), transparency);
        } catch (HeadlessException e) {
        }
        if (bimage == null) {
            int type = BufferedImage.TYPE_INT_RGB;
            if (hasAlpha) {
                type = BufferedImage.TYPE_INT_ARGB;
            }
            bimage = new BufferedImage(image.getWidth(null), image
                    .getHeight(null), type);
        }
        Graphics g = bimage.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();
        return bimage;
    }
}
