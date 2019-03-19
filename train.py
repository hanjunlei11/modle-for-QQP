from model_test import *
from tool import *
from config import *
import tensorflow as tf
from sklearn.metrics import *
s1_char_train,s1_char_test, s2_char_train,s2_char_test = get_char('./QQP/s1.txt','./QQP/s2.txt')
s1_word_train,s1_word_test,s2_word_train,s2_word_test,vector_lines,label_train,label_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test = read_file(s1path='./s1.txt',s2path='./s2.txt',labelpath='./label.txt',re_vector='./vector.txt')
Model = Model(embedding_re=vector_lines)
print('1、构造模型完成')
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Model.losses)
print('2、load data完成')
saver = tf.train.Saver(max_to_keep=3)
miss = [0]*404158
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op)
    # ckpt = tf.train.get_checkpoint_state('./ckpt/')
    # saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始训练')
    max_acc=0
    for i in range(100000):
        batch_s1, batch_s2, batch_label, batch_char_s1, batch_char_s2, batch_s1_len, batch_s2_len, batch_s1_mf, batch_s2_mf, random_int = get_batch(
            batch_size=batch_size, s1=s1_word_train, s2=s2_word_train, s1_len=s1_len_train, s2_len=s2_len_train,
            label=label_train, char_s1=s1_char_train, char_s2=s2_char_train)
        feed_dic = {Model.input_s1: batch_s1, Model.input_s2: batch_s2, Model.batch_label: batch_label,
                    Model.input_char_s1: batch_char_s1,Model.batch_s1_mf:batch_s1_mf,
                    Model.input_char_s2: batch_char_s2,Model.batch_s2_mf:batch_s2_mf,
                    Model.embedding_keep_rate:0.9,Model.keep_rate:0.9,Model.is_traning:True}
        _,rs,loss, acc,prediction=sess.run([opt_op,merged,Model.losses,Model.acc,Model.max_index],feed_dict=feed_dic)
        writer.add_summary(rs, i)
        F1 = f1_score(batch_label, prediction)
        for j in range(batch_size):
            if prediction[j]!=batch_label[j]:
                miss[random_int[j]]+=1
        print(i+1,'次训练 ','loss: ','%.7f'%loss,'acc: ','%.7f'%acc,'max:','%.7f'%max_acc,' F1: ','%.7f'%F1)
        if (i+1)%100==0:
            all_acc = 0
            all_loss = 0
            all_F1 = 0
            for j in range(20):
                batch_s1, batch_s2, batch_label, batch_char_s1, batch_char_s2, batch_s1_len, batch_s2_len, batch_s1_mf, batch_s2_mf, random_int = get_batch(
                    batch_size=batch_size, s1=s1_word_test, s2=s2_word_test, s1_len=s1_len_test, s2_len=s2_len_test,
                    label=label_test, char_s1=s1_char_test, char_s2=s2_char_test)
                feed_dic = {Model.input_s1: batch_s1, Model.input_s2: batch_s2, Model.batch_label: batch_label,
                            Model.input_char_s1: batch_char_s1,Model.batch_s1_mf:batch_s1_mf,
                            Model.input_char_s2: batch_char_s2,Model.batch_s2_mf:batch_s2_mf,
                            Model.embedding_keep_rate:1.0,Model.keep_rate:1.0,Model.is_traning:False}
                loss, acc, prediction= sess.run([Model.losses, Model.acc,Model.max_index], feed_dict=feed_dic)
                F1 = f1_score(batch_label,prediction)
                all_F1+=F1
                all_acc+=acc
                all_loss+=loss
                for k in range(batch_size):
                    if prediction[k] != batch_label[k]:
                        miss[random_int[k]+323326] += 1
                print('test losses: ','%.7f'%loss,' ','accuracy: ','%.7f'%acc,' F1: ','%.7f'%F1)
            all_F1 = all_F1/20.0
            all_acc = all_acc/20.0
            all_loss = all_loss/20.0
            if all_acc > max_acc:
                max_acc = all_acc
                saver.save(sess, save_path='ckpt_test/model.ckpt', global_step=i + 1)
            print('第',int((i+1)/100),'次测试 ','losses: ','%.7f'%all_loss, 'accuracy: ', '%.7f'%all_acc,' F1: ','%.7f'%all_F1)
            # with open('miss.txt','w+',encoding='utf-8') as file:
            #     for k in range(len(miss)):
            #         if k == len(miss):
            #             file.write(str(k + 1) + ' ' + str(miss[k]))
            #         else:
            #             file.write(str(k+1)+' '+str(miss[k])+'\n')