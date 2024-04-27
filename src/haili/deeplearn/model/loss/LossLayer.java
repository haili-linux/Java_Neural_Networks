package haili.deeplearn.model.loss;

import haili.deeplearn.function.Function;
import haili.deeplearn.function.loss.MSELoss;

public class LossLayer {

    public Function loss_function = new MSELoss();

    public float[] gradient(float[] y_pre, float[] y_t){
        float[] grads = new float[y_pre.length];
        for (int i = 0; i < y_t.length; i++)
            grads[i] = loss_function.f_derivative(y_pre[i], y_t[i]);

        return grads;
    }

    public float[] loss_arrays(float[] y_pre, float[] y_t){
        float[] loss = new float[y_pre.length];
        for (int i = 0; i < y_t.length; i++)
            loss[i] = loss_function.f(y_pre[i], y_t[i]);

        return loss;
    }

    public float loss(float[] y_pre, float[] y_t){
        float loss = 0;

        for (int i = 0; i < y_t.length; i++)
            loss += loss_function.f(y_pre[i], y_t[i]);

        loss = loss / y_pre.length;

        return loss;
    }
}
