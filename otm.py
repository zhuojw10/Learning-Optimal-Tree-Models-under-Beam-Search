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
    # training/testing dataset 

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
        # x: b_size * num_fea, numpy array
        # y: b_size, list
        b_size = len(item)
        num_depth = self.depth - self.startSampleDepth
        return_cand, return_cand_mask, return_cand_upstream = [], [], []
        tmp_label = item
        return_user = self.buildTestBatch(user)
        for j in range(num_depth):
            tmp_pos = [np.unique(it) for it in tmp_label]
            tmp_label = [(it-1)/2 for it in tmp_pos]
            tmp_neigh = [tmp_label[i]*4+3-tmp_pos[i] for i in range(b_size)]
            pos_lens = [len(it) for it in tmp_pos]
            maxlen = max(pos_lens)
            cand = np.zeros((b_size, maxlen, 2), dtype=np.int32)
            cand_mask = np.zeros((b_size, maxlen), dtype=np.float32)
            cand_upstream = np.zeros((b_size, maxlen), dtype=np.int32)
            for i in range(b_size):
                cand[i,:pos_lens[i],:] = np.stack([tmp_pos[i], tmp_neigh[i]], axis=-1)
                cand_mask[i,:pos_lens[i]] = 1.
                tmp_upstream = np.cumsum(tmp_label[i][1:]-tmp_label[i][:-1]>0)
                cand_upstream[i,1:pos_lens[i]] = tmp_upstream
                cand_upstream[i,pos_lens[i]:] = cand_upstream[i,pos_lens[i]-1] # to guarantee that map_fn generate tensor with same length
            return_cand.append(cand)
            return_cand_mask.append(cand_mask)
            return_cand_upstream.append(cand_upstream)

        return return_user, return_cand[::-1], return_cand_mask[::-1], return_cand_upstream[::-1]

    def buildTestBatch(self, user):
        num_depth = self.depth - self.startSampleDepth
        return_user = np.zeros((num_depth+1, user.shape[0], user.shape[1]), dtype=np.int32)
        for j in range(num_depth):
            return_user[-(j+1),:,:] = user
            user = (user-1) / 2
            user[user < 0] = 0
        return_user[0,:,:] = user
        return return_user

class OTM():

    def __init__(self, hidden_size, learning_rate, group_size, numTrainBatch, \
            emb_dim=24, num_id=100, num_start=7, num_depth=16, start_depth=8):

        # segment_ids: for making group in user embeddings
        segment_num = []
        tmp = 0
        for i in group_size:
            for j in range(i): # o
                segment_num.append(tmp)
            tmp += 1
        self.segment_ids = tf.constant(segment_num) 
        self.group_size = group_size
        self.emb_dim = emb_dim
        self.num_id = num_id
        self.num_depth = num_depth

        # placeholder
        self.user_input = tf.placeholder(dtype=tf.int32, shape=(None,None,69), name='user_input') # num_depth+1 * b_size * 69
        self.beam_start_input = tf.placeholder(dtype=tf.int32, name='beam_start_input') # b_size * 2^num_start
        self.k = tf.placeholder(dtype=tf.int32, name='k')
        self.train_flag = tf.placeholder(dtype=tf.bool, name='train_flag')
        # placeholder for upstream training
        self.cand, self.cand_mask, self.cand_up = [], [], []
        for i in range(num_depth):
            self.cand.append(tf.placeholder(dtype=tf.int32, name='cand_%d'% i))
            self.cand_mask.append(tf.placeholder(dtype=FLOATX, name='cand_mask_%d' % i))
            self.cand_up.append(tf.placeholder(dtype=tf.int32, name='cand_up_%d'% i))
        self.b_size = tf.shape(self.user_input)[1]

        # trainable variables
        self.emb = tf.get_variable(name='init_emb', shape=(num_id, emb_dim), dtype=FLOATX, initializer=tf.random_normal_initializer)
        
        # model definition (activation functions defined in build_train)
        self.layers = []
        for h_size in hidden_size[:-1]:
            self.layers.append(tf.layers.Dense(h_size, activation=None))
        self.layers.append(tf.layers.Dense(hidden_size[-1], activation=None))

        # train ops
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.loss = self.build_train()
        self.train_op = self.optimizer.minimize(self.loss)

        # test ops:
        self.beam_layer = self.build_test()

    def forward_prop(self, user_f, item_f):
        user_emb = tf.nn.embedding_lookup(self.emb, user_f) # b_size * 69 * emb
        user_window_num = tf.cast((user_f > 0), dtype=tf.float32)
        item_emb = tf.nn.embedding_lookup(self.emb, item_f) # b_size * (2K or 2num_start) * emb_dim
    
        user_emb_poolled = tf.transpose(tf.segment_sum(tf.transpose(user_emb, [1,0,2]), self.segment_ids), [1,0,2])
        user_window_num = tf.reshape(user_window_num, [self.b_size, 69])
        user_window_num_polled = tf.transpose(tf.segment_sum(tf.transpose(user_window_num, [1,0]), self.segment_ids), [1,0])
        user_window_num_polled = tf.stop_gradient(tf.div(tf.ones_like(user_window_num_polled), tf.maximum(1.0, user_window_num_polled)))
        user_emb_poolled = tf.einsum('aij,ai->aij', user_emb_poolled, user_window_num_polled)
        fix_user_emb = tf.tile(tf.expand_dims(user_emb_poolled, axis=1), [1,tf.shape(item_emb)[1],1,1]) # b_size * num_samples * 10 * emb
        fix_user_emb = tf.reverse(fix_user_emb, axis=[2])
        fix_input_emb = tf.concat([fix_user_emb, tf.expand_dims(item_emb, axis=2)], axis=2)
        fix_input_emb = tf.reshape(fix_input_emb, [self.b_size, tf.shape(item_emb)[1], (1+len(self.group_size))*self.emb_dim]) # b_size 

        bs_f_layer = [fix_input_emb]
        layer_id = 0
        for forward_layer in self.layers[:-1]:
            bs_f_layer.append(forward_layer(bs_f_layer[-1]))
            bs_f_layer.append(prelu(bs_f_layer[-1], name='prelu_'+str(layer_id)))
            layer_id += 1
        bs_f_layer.append(self.layers[-1](bs_f_layer[-1]))
        output = bs_f_layer[-1][:,:,1] - bs_f_layer[-1][:,:,0]
        
        return output
        
    def build_train(self):
        beam_layer = [self.beam_start_input]
        loss = []

        # train logic
        # optimal pseudo target upstream
        upstream_score = [self.cand_mask[-1]]
        for_loop = tf.range(self.b_size)
        for i in range(self.num_depth-2,-1,-1):
            output = self.forward_prop(self.user_input[i+2], tf.reshape(self.cand[i+1], [self.b_size, -1]))
            rescale_output = tf.reshape(output, [self.b_size, -1, 2])
            up_score = self.cand_mask[i+1]*tf.one_hot(tf.argmax(rescale_output, -1),depth=2,axis=-1)[:,:,0]*upstream_score[-1] # b_size*maxlen(i)
            maxlen = tf.shape(self.cand[i])[1]-1
            upstream_score.append(tf.map_fn(lambda j: tf.concat([tf.segment_sum(up_score[j], \
                self.cand_up[i+1][j]),tf.zeros((maxlen-self.cand_up[i+1][j,-1],))],0), for_loop, dtype=FLOATX)) # b_size*maxlen(i-1)
        upstream_score = upstream_score[::-1]
        
        # beam search aware training
        for i in range(self.num_depth):
            next_beam = tf.concat([beam_layer[-1]*2+1, beam_layer[-1]*2+2], axis=-1) # b_size * (2K or 2num_start)
            match_mask = tf.cast(tf.equal(next_beam[:,tf.newaxis,:],self.cand[i][:,:,0][:,:,tf.newaxis]),FLOATX) # b_size*maxlen*2min_num
            #### OTM #####
            train_label = tf.stop_gradient(tf.clip_by_value(tf.squeeze(tf.matmul(upstream_score[i][:,tf.newaxis,:],match_mask),axis=1), 0.0,1.0)) # b_size*2min_num
            #### OTM (-OptEst) ####
            #train_label = tf.stop_gradient(tf.clip_by_value(tf.reduce_sum(match_mask,-2), 0.0,1.0)) 
            
            bs_output = self.forward_prop(self.user_input[i+1], next_beam)
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label, logits=bs_output))

            min_num = tf.minimum(self.k, tf.shape(bs_output)[1])
            prob_topk, idx_topk = tf.nn.top_k(bs_output, k=min_num) # should it be clean output or output?
            row_idx = tf.tile(tf.expand_dims(tf.range(self.b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            beam_layer.append(tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num])) # b_size * K

        return tf.concat(loss, axis=1)


    def build_test(self):
        test_beam_layer = [self.beam_start_input]
        for i in range(self.num_depth):
            next_beam = tf.concat([test_beam_layer[-1]*2+1, test_beam_layer[-1]*2+2], axis=-1)
            output = self.forward_prop(self.user_input[i+1], next_beam)
            min_num = tf.minimum(self.k, tf.shape(output)[1])
            prob_topk, idx_topk = tf.nn.top_k(output, k=min_num)
            row_idx = tf.tile(tf.expand_dims(tf.range(self.b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            test_beam_layer.append(tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num])) # b_size * K

        return test_beam_layer


if __name__ == '__main__':

    eval_k = [10,50,100,200]
    hidden_size = [128,64,24,2]
    learning_rate = 1e-3
    group_size = [20,20,10,10,2,2,2,1,1,1]
    Ktest = 400
    Ktrain = 400
    savePath = './result/otm/'    
    loadPath = './data/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    logfile = open(savePath+'log.dat', 'w')
    
    tree = Tree()
    tree.load(loadPath+'tree.dat')
    trainData = Data()
    testData = Data()
    trainData.loadRaw(loadPath+'UserBehavior_train_multi.dat', 'train')  # real train data
    testData.loadRaw(loadPath+'UserBehavior_test.dat', 'test')
    tree.preprocess(trainData, 'train')
    tree.preprocess(testData, 'test')
    print('Data Load Finished, trainData is %d, testData is %d' % (trainData.num, testData.num))
    print('Tree Depth is %d, startSampleDepth is %d' % (tree.depth, tree.startSampleDepth))
    logfile.write('Data Load Finished, trainData is %d, testData is %d\n' % (trainData.num, testData.num))
    logfile.write('Tree Depth is %d, startSampleDepth is %d\n' % (tree.depth, tree.startSampleDepth))

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

    otm_model = OTM(hidden_size=hidden_size, learning_rate=learning_rate, \
        group_size=group_size, numTrainBatch=numTrainBatch, num_id=pow(2,tree.depth), \
        num_start=pow(2,tree.startSampleDepth-1), num_depth=tree.depth-tree.startSampleDepth, \
        start_depth=tree.startSampleDepth)
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
                    beamStart = np.arange(pow(2, tree.startSampleDepth-1)-1,pow(2, tree.startSampleDepth)-1)
                    beamStart = np.repeat(np.expand_dims(beamStart, axis=0), userTest.shape[1], axis=0)
                    feed_pairs = {otm_model.user_input: userTest, \
                        otm_model.beam_start_input: beamStart, otm_model.k: Ktest, otm_model.train_flag: False}
                    beamLayer = sess.run(otm_model.beam_layer, feed_dict=feed_pairs)
                    evaluator.evaluate(beamLayer[-1], j*testBatchSize)
                # output information
                p, r, f, gr, gt = evaluator.output()
                for jj in range(len(eval_k)):
                    print('Top %d : Precision: %f, Recall: %f, Macro F1: %f, Global Recall: %f, Average GT Num: %f' % \
                        (eval_k[jj], p[jj], r[jj], f[jj], gr[jj], gt))
                    logfile.write('Top %d : Precision: %f, Recall: %f, Macro F1: %f, Global Recall: %f, Average GT Num: %f\n' % \
                        (eval_k[jj], p[jj], r[jj], f[jj], gr[jj], gt))

            # train logic
            start_idx = (i*batchSize) % trainData.num
            end_idx = start_idx + batchSize
            trainUser = trainData.user[start_idx:end_idx]
            trainItem = trainData.item[start_idx:end_idx]
            if end_idx > trainData.num:
                trainUser = np.concatenate((trainUser, trainData.user[0:end_idx % trainData.num]), axis=0)
                trainItem = trainItem + trainData.item[0:end_idx % trainData.num]

            # (num_depth+1) * b_size * 69, (num_depth+1) * b_size * tot_num
            userTrain, batchCand, batchCandMask, batchCandUp = tree.buildTrainBatch(trainUser, trainItem)
            beamStart = np.arange(pow(2, tree.startSampleDepth-1)-1,pow(2, tree.startSampleDepth)-1)
            beamStart = np.repeat(np.expand_dims(beamStart, axis=0), userTrain.shape[1], axis=0)
            feed_pairs = {otm_model.user_input: userTrain, otm_model.beam_start_input: beamStart, \
                otm_model.k: Ktrain, otm_model.train_flag: True}
            for j in range(len(otm_model.cand)):
                feed_pairs[otm_model.cand[j]] = batchCand[j]
                feed_pairs[otm_model.cand_mask[j]] = batchCandMask[j]
                feed_pairs[otm_model.cand_up[j]] = batchCandUp[j]
            _, loss = sess.run([otm_model.train_op, otm_model.loss], feed_dict=feed_pairs)

            print('In Batch Iteration %d, training loss is %.4f' % (i, np.mean(loss)))
            logfile.write('In Batch Iteration %d, training loss is %.4f\n' % (i, np.mean(loss)))


    saver.save(sess, savePath + 'model-final')
    logfile.close()
