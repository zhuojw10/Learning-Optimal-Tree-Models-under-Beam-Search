import numpy as np
import tensorflow as tf
import math
import pickle
import random
import time
import scipy.sparse as sp
import os

FLOATX = tf.float32
numUserFea = 69

def prelu(_x, name):
    """
    Parametric ReLU
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('prelu', _x.get_shape()[-1],
                        initializer=tf.constant_initializer(0.25),
                        dtype=tf.float32, trainable=True)
        return tf.maximum(0.0, _x) + alphas * tf.minimum(0.0, _x)

class Evaluator():
    # Evaluating retrieval performance via prec@k, recall@k and F-measure@k

    def __init__(self, gt_item, k, eval_k, leafs):
        self.eval_k = eval_k
        self.hit_num = []
        self.gt_num = []
        self.recall, self.prec, self.f1 = [], [], []
        for jj in range(len(self.eval_k)):
            self.hit_num.append([])
            self.recall.append(0.)
            self.prec.append(0.)
            self.f1.append(0.)
        self.gt_list = []
        self.data_num = len(gt_item)
        self.leafs = leafs
        for i in range(len(gt_item)):
            self.gt_list.append({})
            self.gt_num.append(len(gt_item[i]))
            for j in range(len(gt_item[i])):
                tmp = (gt_item[i][j]-pow(2,k)+1) / pow(2,k)
                if tmp not in self.leafs:
                    continue
                if tmp not in self.gt_list[i]:
                    self.gt_list[i][tmp] = 1
                else:
                    self.gt_list[i][tmp] += 1

    def reset(self):
        self.hit_num = []       
        self.recall, self.prec, self.f1 = [], [], []
        for jj in range(len(eval_k)):
            self.hit_num.append([])
            self.recall.append(0.)
            self.prec.append(0.)
            self.f1.append(0.)
       

    def evaluate(self, pred_item, start_idx):
        for i in range(pred_item.shape[0]):
            for jj in range(len(self.eval_k)):
                tmp_hit_num = 0
                #print(self.gt_list[start_idx+i])
                valid_num = 0
                for j in range(pred_item.shape[1]):
                    if valid_num >= self.eval_k[jj]:
                        break
                    if pred_item[i][j] not in self.leafs:
                        continue
                    valid_num += 1
                    if pred_item[i][j] in self.gt_list[start_idx+i]:
                        tmp_hit_num += self.gt_list[start_idx+i][pred_item[i][j]]
                self.hit_num[jj].append(tmp_hit_num)
                if valid_num > 0:
                    p = tmp_hit_num * 1. / valid_num
                else:
                    p = 0.
                if self.gt_num[start_idx+i] > 0:
                    r = tmp_hit_num * 1. / self.gt_num[start_idx+i]
                else:
                    r = 0.
                if p + r > 0:
                    f = 2 * p * r / (p+r)
                else:
                    f = 0.
                self.prec[jj] += p
                self.recall[jj] += r
                self.f1[jj] += f

    def output(self):
        return_prec = [self.prec[jj]/float(self.data_num) for jj in range(len(self.eval_k))]
        return_recall = [self.recall[jj]/float(self.data_num) for jj in range(len(self.eval_k))]
        return_f1 = [self.f1[jj]/float(self.data_num) for jj in range(len(self.eval_k))]
        return_hit = [sum(self.hit_num[jj])/float(sum(self.gt_num)) for jj in range(len(self.eval_k))]
        return return_prec, return_recall, return_f1, return_hit, sum(self.gt_num) / float(self.data_num)

class Data():

    def __init__(self):
        self.user = []
        self.item = []
        self.num = 0

    def loadRaw(self, path, type='train'):
        with open(path, 'r') as f:
            lines = f.readlines();
            for line in lines:
                tmp_user, tmp_item = [], []
                item = line.split(',')
                for i in range(len(item)):
                    if i < numUserFea:
                        tmp_user.append(int(item[i]))
                    else:
                        tmp_item.append(int(item[i]))
                self.user.append(tmp_user)
                self.item.append(tmp_item)
        self.num = len(self.user)
        self.user = np.array(self.user)
        
    def load(self, path):
        self.user = pickle.load(open(path+'user.dat', 'r'))
        self.item = pickle.load(open(path+'item.dat', 'r'))
        self.num = self.user.shape[0]
        #self.num = len(self.user)

    def save(self, path):
        pickle.dump(self.user, open(path+'user.dat', 'w'))
        pickle.dump(self.item, open(path+'item.dat', 'w'))

class Tree():
    # binary tree
    # id: target(item) index in data.item
    # offset: tree node index, e.g.,
    #        0
    #       / \
    #      1   2
    #     / \ / \
    #     3 4 5 6

    def __init__(self):
        self.id2offset = {}
        self.depth = 0
        self.startSampleDepth = 8

    def load(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.split(':')
                self.id2offset[int(item[0])] = int(item[1])
                if int(math.log(int(item[1])+1, 2))+1 > self.depth:
                    self.depth = int(math.log(int(item[1])+1, 2))+1

    def preprocess(self, data, dtype='train'):
        for i in range(data.num):
            for j in range(len(data.item[i])):
                if data.item[i][j] != 0:
                    data.item[i][j] = self.id2offset[data.item[i][j]]
            for j in range(data.user[i].shape[0]):
                if data.user[i][j] != 0:
                    data.user[i][j] = self.id2offset[data.user[i][j]]
       
    def buildTrainBatch(self, user, item):

        # we split train and test for plt
        b_size = len(user)
        num_depth = self.depth - self.startSampleDepth

        pos_sets, neg_sets = [], []
        tmp_label = [np.array(jj) for jj in item]
        for j in range(num_depth-1,-1,-1):
            pos_tmp = [set(tmp_label[i]) for i in range(b_size)]
            pos_sets = pos_sets + pos_tmp
            tmp_label = [(ite-1)/2 for ite in tmp_label]
            left_cand = [list(ite*2+1) for ite in tmp_label]
            right_cand = [list(ite*2+2) for ite in tmp_label]
            neg_tmp = [set(left_cand[i]).union(set(right_cand[i])).difference(pos_tmp[i]) for i in range(b_size)]
            neg_sets = neg_sets + neg_tmp
        pos_lens = [len(pos_sets[i]) for i in range(len(pos_sets))]
        neg_lens = [len(neg_sets[i]) for i in range(len(neg_sets))]
        totlen = sum(pos_lens)+sum(neg_lens)
        label_idx = np.zeros((totlen,), dtype=np.int32)
        label_val = np.zeros((totlen,), dtype=np.float32)
        label_gather = np.zeros((totlen,), dtype=np.int32)
        offset = 0
        for i in range(len(pos_sets)):
            label_idx[offset:offset+pos_lens[i]] = np.array(list(pos_sets[i]), dtype=np.int32)
            label_idx[offset+pos_lens[i]:offset+pos_lens[i]+neg_lens[i]] = np.array(list(neg_sets[i]), dtype=np.int32)
            label_val[offset:offset+pos_lens[i]] = 1.
            label_gather[offset:offset+pos_lens[i]+neg_lens[i]] = i
            offset += pos_lens[i]+neg_lens[i]

        # user should be tot_len * 69 as well
        return_user = self.buildTestBatch(user)[::-1]
        return return_user, label_idx, label_val, label_gather # should be return depth

    def buildTestBatch(self, user):
        num_depth = self.depth - self.startSampleDepth
        return_user = np.zeros((num_depth, user.shape[0], user.shape[1]), dtype=np.int32)
        for j in range(num_depth-1):
            return_user[-(j+1),:,:] = user
            user = (user-1) / 2
            user[user < 0] = 0
        return_user[0,:,:] = user
        return return_user

class PLT():

    def __init__(self, hidden_size, learning_rate, group_size, numTrainBatch, \
            emb_dim=24, num_id=100, num_start=7, num_depth=16, start_depth=8, K=200):

        print(tf.__version__)

        # segment_ids: for making group in user embeddings
        segment_num = []
        tmp = 0
        for i in group_size:
            for j in range(i): # o
                segment_num.append(tmp)
            tmp += 1
        self.segment_ids = tf.constant(segment_num) 
        self.emb_dim = emb_dim
        self.start_depth = start_depth
        self.num_depth = num_depth

        # placeholder
        self.user_input = tf.placeholder(dtype=tf.int32, shape=(None,None,69), name='user_input') # (num_depth*b_size) * 69
        self.item_idx_input = tf.placeholder(dtype=tf.int32, shape=(None,), name='item_idx_input')
        self.item_val_input = tf.placeholder(dtype=FLOATX, shape=(None,), name='item_val_input')
        self.item_gather_input = tf.placeholder(dtype=tf.int32, shape=(None,), name='item_gather_input')
        self.train_flag = tf.placeholder(dtype=tf.bool, name='train_flag')
        self.b_size = tf.shape(self.user_input)[1]
       
        self.emb = tf.get_variable(name='init_emb', shape=(num_id, self.emb_dim), dtype=FLOATX, trainable=True)
        self.user_emb = tf.nn.embedding_lookup(self.emb, self.user_input)
        self.layers = []
        for h_size in hidden_size[:-1]:
            self.layers.append(tf.layers.Dense(h_size, activation=None))
        self.layers.append(tf.layers.Dense(hidden_size[-1], activation=None, trainable=True))
        
        # train logic
        self.loss = self.build_train()
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        # test logic
        self.beam_layer = self.build_test()

    
    def build_train(self):

        def combine_user_item_gather(user, user_win, item, gather_id):
            # user: ? * 69 * emb_dim, item: ? * emb_dim
            user_poolled = tf.transpose(tf.segment_sum(tf.transpose(user, [1,0,2]), \
                self.segment_ids), [1,0,2]) #(num_depth*b_size) * 10 * emb_dim
            user_win_polled = tf.transpose(tf.segment_sum(tf.transpose(user_win, [1,0]), self.segment_ids), [1,0])
            user_win_polled = tf.stop_gradient(tf.div(tf.ones_like(user_win_polled), tf.maximum(1.0, user_win_polled)))
            user_poolled = tf.einsum('aij,ai->aij', user_poolled, user_win_polled)
            user_gather = tf.gather(user_poolled, gather_id, axis=0)
            combine_user_item = tf.concat([user_gather, item[:,None,:]], axis=1)
            return tf.reshape(combine_user_item, [-1, 11*self.emb_dim])

        train_user_emb = tf.reshape(self.user_emb, [-1, 69, self.emb_dim])
        train_user_win = tf.cast((self.user_input > 0), dtype=tf.float32)
        train_user_win = tf.reshape(train_user_win, [-1, 69])
        item_emb = tf.nn.embedding_lookup(self.emb, self.item_idx_input)
        train_layer = [combine_user_item_gather(train_user_emb, train_user_win, item_emb, self.item_gather_input)]
        layer_id = 0
        for f_layer in self.layers[:-1]:
            train_layer.append(f_layer(train_layer[-1]))
            train_layer.append(prelu(train_layer[-1], name='prelu_'+str(layer_id)))
            layer_id +=1
        train_layer.append(self.layers[-1](train_layer[-1]))

        output = train_layer[-1][:,1] - train_layer[-1][:,0]
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=self.item_val_input, logits=output)

    def build_test(self):

        def combine_user_item(user, user_win, item):
            # user: b_size * 69 * emb_dim, item: b_size * num_samples * emb_dim
            user_poolled = tf.transpose(tf.segment_sum(tf.transpose(user, [1,0,2]), self.segment_ids), [1,0,2]) # b_size * 10 * emb_dim
            user_win_polled = tf.transpose(tf.segment_sum(tf.transpose(user_win, [1,0]), self.segment_ids), [1,0])
            user_win_polled = tf.stop_gradient(tf.div(tf.ones_like(user_win_polled), tf.maximum(1.0, user_win_polled)))
            user_poolled = tf.einsum('aij,ai->aij', user_poolled, user_win_polled)
            return tf.reshape(tf.concat([tf.tile(user_poolled[:,None,:,:],[1,tf.shape(item)[1],1,1]), \
                item[:,:,None,:]], axis=2), [self.b_size,-1,11*self.emb_dim]) # b_size * num_samples * 11 * emb_dim

        test_beam_layer = [tf.tile(tf.range(pow(2,self.start_depth-1)-1, pow(2,self.start_depth)-1)[None,:],[self.b_size,1])]
        bs_layer = []
        prod_score = [tf.ones_like(test_beam_layer[-1], dtype=FLOATX)]
        for i in range(self.num_depth):
            next_beam = tf.concat([test_beam_layer[-1]*2+1, test_beam_layer[-1]*2+2], axis=-1) # b_size * (2K or 2num_start)
            cur_score = tf.concat([prod_score[-1], prod_score[-1]], axis=-1)
            bs_emb = tf.nn.embedding_lookup(self.emb, next_beam)
            user_win = tf.cast((self.user_input[i] > 0), dtype=tf.float32)
            user_win = tf.reshape(user_win, [-1, 69])
            bs_layer.append(combine_user_item(self.user_emb[i], user_win, bs_emb))
            layer_id = 0
            for f_layer in self.layers[:-1]:
                bs_layer.append(f_layer(bs_layer[-1]))
                bs_layer.append(prelu(bs_layer[-1], name='prelu_'+str(layer_id)))
                layer_id += 1
            bs_layer.append(self.layers[-1](bs_layer[-1]))
            bs_output = tf.nn.sigmoid(bs_layer[-1][:,:,1]-bs_layer[-1][:,:,0]) * cur_score

            min_num = tf.minimum(tf.shape(bs_output)[1], K) 
            prob_topk, idx_topk = tf.nn.top_k(bs_output, k=min_num)
            row_idx = tf.tile(tf.expand_dims(tf.range(self.b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            test_beam_layer.append(tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num])) # b_size * K
            prod_score.append(tf.reshape(tf.gather_nd(bs_output, all_idx), [-1, min_num]))
        
        return test_beam_layer

if __name__ == '__main__':

    eval_k = [10,50,100,200]
    hidden_size = [128,64,24,2]
    learning_rate = 1e-3
    group_size = [20,20,10,10,2,2,2,1,1,1]
    K = 400
    choice = 'loadRaw'
    savePath = './result/plt/'
    loadPath = './data/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    logfile = open(savePath+'log.dat', 'w')
    
    tree = Tree()
    tree.load(loadPath+'tree.dat')

    trainData = Data()
    testData = Data()
    trainData.loadRaw(loadPath+'UserBehavior_train_multi.dat', 'train')
    testData.loadRaw(loadPath+'UserBehavior_test.dat', 'test')
    tree.preprocess(trainData, 'train')
    tree.preprocess(testData, 'test')
    print('Data Load Finished, trainData is %d, testData is %d' % (trainData.num, testData.num))
    print('Tree Depth is %d, startSampleDepth is %d' % (tree.depth, tree.startSampleDepth))
    logfile.write('Data Load Finished, trainData is %d, testData is %d\n' % (trainData.num, testData.num))
    logfile.write('Tree Depth is %d, startSampleDepth is %d\n' % (tree.depth, tree.startSampleDepth))
 
    #no need for epoch here
    numEpoch = 10
    batchSize = 200 
    testBatchSize = 500
    testPer = 100
    numTrainBatch = numEpoch * trainData.num / batchSize + 1
    numTestBatch = testData.num / testBatchSize + 1 
    print('Total batch iteration is %d' % numTrainBatch)
    logfile.write('Total batch iteration is %d\n' % numTrainBatch)

    leafs = set([]) # filter for leaf nodes without correpsonding target id.
    with open(loadPath+'tree.dat', 'r') as f:
        for line in f.readlines():
            leafs.add(int(line.strip().split(':')[1]))
    print('Leaf num: %d' % len(leafs))
    logfile.write('Leaf num: %d\n' % len(leafs))
    evaluator = Evaluator(testData.item, 0, eval_k, leafs)

    plt_model = PLT(hidden_size=hidden_size, learning_rate=learning_rate, \
            group_size=group_size, numTrainBatch=numTrainBatch, num_id=pow(2,tree.depth), \
            num_start=pow(2,tree.startSampleDepth-1), num_depth=tree.depth-tree.startSampleDepth, \
            start_depth=tree.startSampleDepth, K=K)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
                
        for i in range(numTrainBatch):

            # test info output
            if i % testPer == 0:
                evaluator.reset()
                for j in range(numTestBatch):
                    if j*testBatchSize >= testData.num:
                        continue
                    userTest = tree.buildTestBatch(testData.user[j*testBatchSize:(j+1)*testBatchSize])
                    feed_pairs = {plt_model.user_input: userTest, plt_model.train_flag: False}
                    beamLayer = sess.run(plt_model.beam_layer, feed_dict=feed_pairs)
                    evaluator.evaluate(beamLayer[-1], j*testBatchSize)                   
                # output information
                print('In Batch Iteration %d,' % i)
                logfile.write('In Batch Iteration %d,' % i)
                p, r, f, gr, gt = evaluator.output()
                for jj in range(len(eval_k)):
                    print('Top %d : Precision: %f, Recall: %f, Macro F1: %f, Global Recall: %f, Average GT Num: %f' % \
                        (eval_k[jj], p[jj], r[jj], f[jj], gr[jj], gt))
                    logfile.write('Top %d : Precision: %f, Recall: %f, Macro F1: %f, Global Recall: %f, Average GT Num: %f\n' % \
                        (eval_k[jj], p[jj], r[jj], f[jj], gr[jj], gt))
            else:
                # build batch
                start_idx = (i*batchSize) % trainData.num
                end_idx = start_idx + batchSize
                trainUser = trainData.user[start_idx:end_idx]
                trainItem = trainData.item[start_idx:end_idx]
                if end_idx > trainData.num:
                    trainUser = np.concatenate((trainUser, trainData.user[0:end_idx % trainData.num]), axis=0)
                    trainItem = trainItem + trainData.item[0:end_idx % trainData.num]

                userTrain, itemIdx, itemVal, itemGather = tree.buildTrainBatch(trainUser, trainItem)
                feed_pairs = {plt_model.user_input: userTrain, plt_model.item_idx_input: itemIdx, \
                    plt_model.item_val_input: itemVal, plt_model.item_gather_input: itemGather}
                _, loss = sess.run([plt_model.train_op, plt_model.loss], feed_dict=feed_pairs)

                print('In Batch Iteration %d, training loss is %.4f' % (i, np.mean(loss)))
                logfile.write('In Batch Iteration %d, training loss is %.4f\n' % (i, np.mean(loss)))


    logfile.close()

