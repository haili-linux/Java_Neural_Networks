
#
# 本函数仅可保存通过tf.keras.models.Sequential(）创建的全连接网络模型, 即仅含Dense层的模型
# 通过该函数保存的模型，可以用使用本项目中的模型读取
# 支持的激活函数为 activation_id[sigmoid=1, tanh=2, rule=3, leaky_rule=4]
# 以下是参数说明：
#   model: tf.keras.models.Sequential(）创建的全连接网络模型
#   filename: 保存后的文件名
#   activation_id：神经网络每层的激活函数id数组
# 示例：saveInFile(model,'test3.txt',[4,4,4,4,4,4,1])
# activation_id[sigmoid=1, tanh=2, rule=3, leaky_rule=4]

def saveInFile(model, filename, activation_id):
    w0 = tf.transpose(model.weights[0])
    in_vector = len(w0[0])
    out_vector = len(tf.transpose(model.weights[len(model.weights) - 1]))

    hidden_n = 0
    hidden_arf_str = '['
    i = 0
    while i < len(model.weights) - 2:
        hidden_n = hidden_n + 1
        i += 2
        l = len(tf.transpose(model.weights[i])[0].numpy())
        hidden_arf_str += str(l)
        if i < len(model.weights) - 2:
            hidden_arf_str += ', '
    hidden_arf_str += ']'
    # print(hidden_n)
    # print(hidden_arf_str)

    file = open(file=filename, mode='w')
    file.write('explain:null\n')
    file.write('in_vector:' + str(in_vector) + '\n')
    file.write('out_vector:' + str(out_vector) + '\n')
    file.write('nl:1.0E-4\n')
    file.write('hid_n:length:' + str(hidden_n) + ' ' + hidden_arf_str + '\n')
    file.write('de:0.0\n')
    file.write('_in_max:1.0\n')
    file.write('_out_max:1.0\n')
    file.write('ACT_FUCTION:' + str(activation_id[0]) + '\n')
    file.write('LOSS_FUCTION:10\n')

    model_len = len(model.weights)
    out_W = tf.transpose(model.weights[model_len - 2])
    out_d = tf.transpose(model.weights[model_len - 1])

    i = 0
    while i < len(out_d):
        var0 = 'outputNeuer[' + str(i) + '].act_fuction_id:' + str(activation_id[len(activation_id)-1]) + '\n' + 'outputNeuer[' + str(i) + \
               '].d:' + str(out_d[i].numpy()) + '\n'
        file.write(var0)
        # file.write('outputNeuer[' + str(i) + '].act_fuction_id:' + str(activation_id) + '\n')
        # file.write('outputNeuer[' + str(i) + '].d:' + str(out_d[i].numpy()) + '\n')
        var0 = ''
        j = 0
        while j < len(out_W[i]):
            # file.write('outputNeuer[' + str(i) + '].w' + str(j) + ':' + str(out_W[i][j].numpy()) + '\n')
            var0 += 'outputNeuer[' + str(i) + '].w' + str(j) + ':' + str(out_W[i][j].numpy()) + '\n'
            if len(var0) >= 2048:
                file.write(var0)
                var0 = ''
            j += 1
        i += 1

        file.write(var0)

    arf_n = 0
    while arf_n < len(model.weights) - 2:
        w = tf.transpose(model.weights[arf_n])
        # 当层神经元个数
        layer_n = len(model.weights[arf_n + 1].numpy())

        i = 0
        while i < layer_n:

            var0 = 'hiddenNeuer[' + str(int(arf_n/2)) + '][' + str(i) + '].act_fuction_id:' + \
                   str(activation_id[int(arf_n/2)]) + '\n' + \
                   'hiddenNeuer[' + str(int(arf_n/2)) + '][' + str(i) + '].d:' + \
                   str(model.weights[arf_n + 1].numpy()[i]) + '\n'
            file.write(var0)

            var0 =''
            wi = w[i]
            j = 0
            while j < len(wi):
                var0 += 'hiddenNeuer[' + str(arf_n/2) + '][' + str(i) + '].w' + str(j) + ':' + str(wi[j].numpy()) + '\n'
                if len(var0) >= 2048:
                    file.write(var0)
                    var0 = ''
                j += 1

            file.write(var0)

            i += 1

        arf_n += 2
    file.close()