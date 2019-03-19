import tensorflow as tf
from config import *
from Multi_Head_Attention import *

class Model():
    def __init__(self,embedding_re):
        self.batch_size = batch_size
        self.embedding_size = embadding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_len = batch_len
        self.word_length = word_length
        self.char_size = char_size
        self.embedding_char_size = embadding_char_size
        self.L2_reregularizer = tf.contrib.layers.l2_regularizer(10e-6)
        self.L2_reregularizetion = 0.01
        with tf.name_scope("input"):
            self.input_s1 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.batch_len))
            self.input_s2 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.batch_len))
            self.batch_label = tf.placeholder(dtype=tf.int64,shape=(self.batch_size))
            self.input_char_s1 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.batch_len,self.word_length))
            self.input_char_s2 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.batch_len,self.word_length))
            self.embedding_keep_rate = tf.placeholder(dtype=tf.float32,shape=(None))
            self.keep_rate = tf.placeholder(dtype=tf.float32,shape=(None))
            self.is_traning = tf.placeholder(dtype=tf.bool,shape=(None))
            self.batch_s1_mf = tf.placeholder(dtype=tf.float32,shape=(self.batch_size,self.batch_len))
            self.batch_s2_mf = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.batch_len))

        with tf.name_scope("embedding_layer_re"):
            embedding_table_re = tf.Variable(embedding_re,trainable=True,name='re_train_embedding')
            s1_matrix_fix = tf.nn.embedding_lookup(embedding_table_re,self.input_s1)
            s1_matrix_fix = tf.nn.dropout(s1_matrix_fix,keep_prob=self.embedding_keep_rate)
            s2_matrix_fix = tf.nn.embedding_lookup(embedding_table_re,self.input_s2)
            s2_matrix_fix = tf.nn.dropout(s2_matrix_fix, keep_prob=self.embedding_keep_rate)

        with tf.name_scope('embedding_layer_train'):
            embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], dtype=tf.float32),
                                          trainable=True, name='word_embedding')
            s1_matrix_tr = tf.nn.embedding_lookup(embedding_table, self.input_s1)
            s1_matrix_tr = tf.nn.dropout(s1_matrix_tr, keep_prob=self.embedding_keep_rate)
            s2_matrix_tr = tf.nn.embedding_lookup(embedding_table, self.input_s2)
            s2_matrix_tr = tf.nn.dropout(s2_matrix_tr,keep_prob=self.embedding_keep_rate)

        with tf.name_scope("char_embedding_conv"):
            embedding_char = tf.Variable(tf.random_uniform([self.char_size, self.embedding_char_size], dtype=tf.float32),trainable=True,name='char_embedding')
            s1_char_embedding = tf.nn.embedding_lookup(embedding_char,self.input_char_s1)
            s2_char_embedding = tf.nn.embedding_lookup(embedding_char,self.input_char_s2)
            self.s1_char_conv_out = tf.nn.dropout(s1_char_embedding, keep_prob=self.embedding_keep_rate)
            self.s2_char_conv_out = tf.nn.dropout(s2_char_embedding, keep_prob=self.embedding_keep_rate)
            self.s1_char_conv_out = self.conv2D(inputs=self.s1_char_conv_out,kernel_shape=[1,5,100,100],strides=[1,1,1,1],padding='VALID',kernel_name='char_kernel1')
            self.s1_char_conv_out = tf.layers.max_pooling2d(inputs=self.s1_char_conv_out,pool_size=[1,12],strides=[1,1])
            self.s1_char_conv_out = tf.reduce_mean(self.s1_char_conv_out, axis=2)
            self.s2_char_conv_out = self.conv2D(inputs=self.s2_char_conv_out, kernel_shape=[1, 5, 100, 100],strides=[1, 1, 1, 1], padding='VALID', kernel_name='char_kernel2')
            self.s2_char_conv_out = tf.layers.max_pooling2d(inputs=self.s2_char_conv_out, pool_size=[1, 12],strides=[1, 1])
            self.s2_char_conv_out = tf.reduce_mean(self.s2_char_conv_out, axis=2)

        with tf.name_scope('concat_input'):
            self.batch_s1_0 = tf.concat([s1_matrix_fix, s1_matrix_tr, self.s1_char_conv_out,tf.expand_dims(self.batch_s1_mf,axis=-1)], axis=-1)
            self.batch_s2_0 = tf.concat([s2_matrix_fix, s2_matrix_tr, self.s2_char_conv_out,tf.expand_dims(self.batch_s2_mf,axis=-1)], axis=-1)

        with tf.name_scope('LSTM_attention_layer'):
            self.lstm_s1_1, self.lstm_s2_1, self.batch_s1_1, self.batch_s2_1 = self.Dynamic_LSTM(self.batch_s1_0,self.batch_s2_0,self.batch_s1_0, self.batch_s2_0, name='1',use_position=True)
            self.lstm_s1_2, self.lstm_s2_2, self.batch_s1_2, self.batch_s2_2 = self.Dynamic_LSTM(self.lstm_s1_1, self.lstm_s2_1, self.batch_s1_1, self.batch_s2_1, name='2')
            self.lstm_s1_3, self.lstm_s2_3, self.batch_s1_3, self.batch_s2_3 = self.Dynamic_LSTM(self.lstm_s1_2, self.lstm_s2_2, self.batch_s1_2, self.batch_s2_2, name='3')
            self.lstm_s1_4, self.lstm_s2_4, self.batch_s1_4, self.batch_s2_4 = self.Dynamic_LSTM(self.lstm_s1_3, self.lstm_s2_3, self.batch_s1_3, self.batch_s2_3, name='4')
            self.lstm_s1_5, self.lstm_s2_5, self.batch_s1_5, self.batch_s2_5 = self.Dynamic_LSTM(self.lstm_s1_4, self.lstm_s2_4, self.batch_s1_4, self.batch_s2_4, name='5')
            self.lstm_s1_6, self.lstm_s2_6, self.batch_s1_6, self.batch_s2_6 = self.Dynamic_LSTM(self.lstm_s1_5, self.lstm_s2_5, self.batch_s1_5, self.batch_s2_5, name='6')
            self.batch_s1_6 = tf.reshape(tf.layers.max_pooling1d(inputs=self.batch_s1_6, pool_size=21, strides=1),shape=[self.batch_size,1000])
            self.batch_s2_6 = tf.reshape(tf.layers.max_pooling1d(inputs=self.batch_s2_6, pool_size=21, strides=1),shape=[self.batch_size, 1000])
            self.concat = tf.concat([self.batch_s1_6,self.batch_s2_6],axis=-1)

        with tf.name_scope('dense_and_softmax_layer'):
            self.Dense_1 = self.fully_conacation(input=self.concat,haddin_size=200,keep_rate=self.keep_rate,activation='None')
            self.Dense_2 = self.fully_conacation(input=self.Dense_1,haddin_size=20,keep_rate=self.keep_rate,activation='None')
            self.Dense_4 = self.fully_conacation(input=self.Dense_2,haddin_size=2,activation='sigmoid')

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_label,logits=self.Dense_4),axis=-1)
            tf.add_to_collection('losses',self.loss)
            regularization_loss = self.L2_reregularizetion * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.losses = tf.add_n(tf.get_collection('losses'))+regularization_loss
            tf.summary.scalar('loss', self.losses)

        with tf.name_scope('acc'):
            self.max_index = tf.argmax(self.Dense_4,axis=1)
            cast_value = tf.cast(tf.equal(self.max_index,self.batch_label),dtype=tf.float32)
            self.acc = tf.reduce_mean(cast_value,axis=-1)
            tf.summary.scalar('acc', self.acc)

    def self_attention(self,s):
        temp5 = tf.Variable(tf.random_normal(shape=(batch_size, batch_len, 200), mean=0, stddev=2, dtype=tf.float32))
        for i in range(s.shape[1]):
            temp = tf.slice(s, [0, i, 0], [-1, 1, -1])
            temp4 = tf.reduce_mean(temp, axis=1)
            for j in range(s.shape[1]):
                temp1 = tf.slice(s, [0, j, 0], [-1, 1, -1])
                temp2 = tf.multiply(temp, temp1)
                temp3 = tf.concat([temp, temp1, temp2], axis=-1)
                if j == 0:
                    temp4 = temp3
                else:
                    temp4 = tf.concat([temp4, temp3], axis=1)
            temp4 = tf.expand_dims(temp4, axis=1)
            if i == 0:
                temp5 = temp4
            else:
                temp5 = tf.concat([temp5,temp4], axis=1)
        attention_W = tf.Variable(tf.random_normal(shape=(1, 600), mean=0, stddev=2, dtype=tf.float32))
        attention_A = tf.reduce_sum(tf.multiply(attention_W, temp5), keep_dims=False, axis=-1)
        attention_A = tf.nn.softmax(attention_A, dim=-1)
        attention_P = tf.Variable(tf.random_normal(shape=(self.batch_size, 1, 400), mean=1, stddev=2, dtype=tf.float32))
        for i in range(attention_A.shape[-1]):
            temp = tf.slice(attention_A, [0, 0, i], [-1, -1, 1])
            temp2 = tf.reduce_sum(tf.multiply(temp, s), keep_dims=True, axis=1)
            if i == 0:
                attention_P = temp2
            else:
                attention_P = tf.concat([attention_P, temp2], axis=1)
        return attention_P

    def co_attention(self,s1,s2):
        # attention构造
        # 计算cosin匹配矩阵
        matrix_1 = tf.matmul(s1, tf.transpose(s2, perm=[0, 2, 1]))
        matrix_2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s1), axis=-1)), axis=-1)
        matrix_3 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s2), axis=-1)), axis=1)
        cosin_matrix_s1 = tf.div(matrix_1, tf.matmul(matrix_2, matrix_3))
        # 计算相似矩阵权重
        cosin_matrix_s1 = tf.nn.softmax(cosin_matrix_s1, dim=-1)
        cosin_matrix_s2 = tf.transpose(cosin_matrix_s1, perm=[0, 2, 1])
        cosin_matrix_s2 = tf.nn.softmax(cosin_matrix_s2, dim=-1)
        a_s2 = tf.matmul(cosin_matrix_s1, s2)
        a_s1 = tf.matmul(cosin_matrix_s2, s1)

        return a_s1, a_s2

    def fully_conacation(self,input,haddin_size,keep_rate=1.0,activation='leaky_relu'):
        dense_out = tf.layers.dense(inputs=input, units=haddin_size,kernel_regularizer=self.L2_reregularizer)
        dense_out = tf.layers.batch_normalization(inputs=dense_out,training=self.is_traning)
        if activation == 'relu':
            dense_relu = tf.nn.relu(dense_out)
            dense_relu = tf.nn.dropout(dense_relu,keep_prob=keep_rate)
            return dense_relu
        elif activation == 'leaky_relu':
            dense_relu = tf.nn.leaky_relu(dense_out)
            dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
            return dense_relu
        elif activation == 'sigmoid':
            dense_relu = tf.nn.sigmoid(dense_out)
            dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
            return dense_relu
        elif activation == 'None':
            dense_relu = tf.nn.dropout(dense_out, keep_prob=keep_rate)
            return dense_relu

    def conv2D(self,inputs,kernel_shape,strides,padding,kernel_name,activation='leaky_relu',dropuot_rate=None):
        kernel = tf.get_variable(dtype=tf.float32,shape=kernel_shape,name=kernel_name,regularizer=self.L2_reregularizer)
        conv_output = tf.nn.conv2d(input=inputs,filter=kernel,strides=strides,padding=padding)
        conv_output = tf.layers.batch_normalization(inputs=conv_output,training=self.is_traning)
        if activation == 'relu':
            conv_output = tf.nn.relu(conv_output)
        elif activation == 'leaky_relu':
            conv_output = tf.nn.leaky_relu(conv_output)
        if dropuot_rate is not None:
            conv_output = tf.nn.dropout(conv_output,keep_prob=dropuot_rate)
        return conv_output

    def conv1D(self,inputs,kernel_shape,bias_shape,strides,padding,kernel_name,bias_name,activation='leaky_relu',dropuot_rate=None):
        kernel = tf.get_variable(dtype=tf.float32, shape=kernel_shape, name=kernel_name,regularizer=self.L2_reregularizer)
        conv_output = tf.nn.conv1d(value=inputs, filters=kernel, stride=strides, padding=padding)
        conv_output = tf.layers.batch_normalization(inputs=conv_output, training=self.is_traning)
        if activation is 'relu':
            conv_output = tf.nn.relu(conv_output)
        elif activation is 'leaky_relu':
            conv_output = tf.nn.leaky_relu(conv_output)
        if dropuot_rate is not None:
            conv_output = tf.nn.dropout(conv_output, keep_prob=dropuot_rate)
        return conv_output

    def dense_block(self,input,nb_layer,strides,padding,name):
        x = input
        for i in range(nb_layer):
            conv_out = self.conv2D(inputs=x,kernel_shape=[3,3,x.shape[3],32],strides=strides,padding=padding,dropuot_rate=self.keep_rate,kernel_name=name+'kernel'+str(i))
            x = tf.concat([x,conv_out],axis=-1)
        return x

    def transition_block(self,input,output_channel,kernel_name):
        x = self.conv2D(inputs=input,kernel_shape=[1,1,input.shape[3],output_channel],strides=[1,1,1,1],padding='VALID',dropuot_rate=self.keep_rate,kernel_name=kernel_name+'kernel')
        x_output = tf.nn.max_pool(value=x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        return x_output

    def Dynamic_LSTM(self,s1_f,s2_f,input_s1,input_s2,name,use_position=False):
        with tf.variable_scope("lst_"+str(name)+"_1"):
            cell_f_1 = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,reuse=tf.AUTO_REUSE)
            cell_b_1 = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,reuse=tf.AUTO_REUSE)
            lstm_output_s1, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                       inputs=input_s1,
                                                                       dtype=tf.float32)
            lstm_fw_s1, lstm_bw_s1 = lstm_output_s1
            lstm_output_s1 = tf.concat([lstm_fw_s1, lstm_bw_s1], axis=-1)
        # with tf.variable_scope("lst_"+str(name)+"_2"):
            lstm_output_s2, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                       inputs=input_s2,
                                                                       dtype=tf.float32)
            lstm_fw_s2, lstm_bw_s2 = lstm_output_s2
            lstm_output_s2 = tf.concat([lstm_fw_s2, lstm_bw_s2], axis=-1)
        attention_s1,attention_s2 = self.co_attention(lstm_output_s1,lstm_output_s2)
        concat_s1 = tf.concat([s1_f, lstm_output_s1,attention_s1], axis=-1)
        concat_s2 = tf.concat([s2_f, lstm_output_s2,attention_s2], axis=-1)
        encoder_s1_1 = encoder(inputs=concat_s1, embedding_size=self.hidden_size, nb_layers=1, nb_head=4,
                               size_per_head=50, training=self.is_traning, keep_rate=self.keep_rate,
                               activition='leaky_relu',use_position=use_position)
        encoder_s2_1 = encoder(inputs=concat_s2, embedding_size=self.hidden_size, nb_layers=1, nb_head=4,
                               size_per_head=50, training=self.is_traning, keep_rate=self.keep_rate,
                               activition='leaky_relu',use_position=use_position)
        return lstm_output_s1,lstm_output_s2,encoder_s1_1,encoder_s2_1


# model = Model_DRCN()
# init_op=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     sess.run(model.s2_char_conv_out)