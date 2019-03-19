from model import *
from tool import *
from config import *
import tensorflow as tf
s1_char_train,s1_char_test, s2_char_train,s2_char_test = get_char('./first questions.csv')
s1_word_train,s1_word_test,s2_word_train,s2_word_test,vector_lines,label_train,label_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test = read_file(s1path='./s1.txt',s2path='./s2.txt',labelpath='./label.txt',re_vector='./vector.txt')
Model = Model_DCMM(embedding_re=vector_lines)
print('1、构造模型完成')
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    opt_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(Model.losses)
print('2、load data完成')
saver = tf.train.Saver(max_to_keep=3)
# init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    # sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state('./ckpt1/')
    saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始训练')
    max_acc=0
    for i in range(20000):
        batch_s1, batch_s2, batch_label,batch_char_s1, batch_char_s2,batch_s1_len,batch_s2_len= get_batch(
            batch_size=batch_size, s1=s1_word_train, s2=s2_word_train,s1_len=s1_len_train,s2_len=s2_len_train, label=label_train, char_s1=s1_char_train, char_s2=s2_char_train)
        feed_dic = {Model.input_s1: batch_s1, Model.input_s2: batch_s2, Model.batch_label: batch_label,
                    Model.input_char_s1: batch_char_s1,Model.batch_s1_len:batch_s1_len,
                    Model.input_char_s2: batch_char_s2,Model.batch_s2_len:batch_s2_len,
                    Model.embedding_keep_rate:0.9,Model.keep_rate:0.8,Model.is_traning:True}
        _,loss, acc=sess.run([opt_op,Model.losses,Model.acc],feed_dict=feed_dic)
        print('第',i+1,'次训练 ','loss: ',loss,'  ','acc: ',acc)
        if (i+1)%100==0:
            all_acc = 0
            all_loss = 0
            for j in range(10):
                batch_s1, batch_s2, batch_label, batch_char_s1, batch_char_s2,batch_s1_len,batch_s2_len= get_batch(
                    batch_size=batch_size, s1=s1_word_test, s2=s2_word_test,s1_len=s1_len_test,s2_len=s2_len_test, label=label_test, char_s1=s1_char_test, char_s2=s2_char_test)
                feed_dic = {Model.input_s1: batch_s1, Model.input_s2: batch_s2, Model.batch_label: batch_label,
                            Model.input_char_s1: batch_char_s1,Model.batch_s1_len:batch_s1_len,
                            Model.input_char_s2: batch_char_s2,Model.batch_s2_len:batch_s2_len,
                            Model.embedding_keep_rate:1.0,Model.keep_rate:1.0,Model.is_traning:False}
                loss, acc = sess.run([Model.losses, Model.acc], feed_dict=feed_dic)
                all_acc+=acc
                all_loss+=loss
                print('test losses: ',loss,' ','accuracy: ',acc)
            all_acc = all_acc/10.0
            all_loss = all_loss/10.0
            if all_acc > max_acc:
                max_acc = all_acc
                saver.save(sess, save_path='ckpt2/model.ckpt', global_step=i + 1)
            print('第',int((i+1)/100),'次测试 ','losses: ',all_loss, 'accuracy: ', all_acc)