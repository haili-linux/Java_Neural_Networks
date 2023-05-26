package haili.deeplearn.utils;

import java.util.Arrays;

public class TrainData {
    public float[] train_x;
    public float[] train_y;

    public TrainData(float[] train_x, float[] train_y) {
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
