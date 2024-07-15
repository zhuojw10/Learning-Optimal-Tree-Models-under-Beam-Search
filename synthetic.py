import numpy as np
import tensorflow as tf
import math
import pickle
import random
import time
import scipy.sparse as sp
from scipy.stats import beta

import argparse
import os

FLOATX = tf.float32

class Evaluator():

    def __init__(self, tree, gt_prob, gt_item, k, eval_k):
        self.hit_num = []
        self.gt_num = []
        self.recall, self.prec, self.f1, self.prob = [], [], [], []
        for jj in range(len(eval_k)):
            self.hit_num.append([])
            self.recall.append(0.)
            self.prec.append(0.)
            self.f1.append(0.)
            self.prob.append(0.)
        self.gt_list = []
        self.prob_list = []
        self.data_num = len(gt_item)
        for i in range(len(gt_item)):
            self.gt_list.append({})
            self.prob_list.append(dict(zip(pow(tree.arity,tree.depth-1)-1+np.arange(pow(tree.arity,tree.depth-1)),gt_prob[i])))
            self.gt_num.append(len(gt_item[i]))
            for j in range(len(gt_item[i])):
                tmp = (gt_item[i][j]-pow(2,k)+1) / pow(2,k)
                if tmp not in self.gt_list[i]:
                    self.gt_list[i][tmp] = 1
                else:
                    self.gt_list[i][tmp] += 1

    def reset(self):
        self.hit_num = []        
        self.recall, self.prec, self.f1, self.prob = [], [], [], []
        for jj in range(len(eval_k)):
            self.hit_num.append([])
            self.recall.append(0.)
            self.prec.append(0.)
            self.f1.append(0.)     
            self.prob.append(0.)

    def evaluate(self, pred_item, start_idx):
        for i in range(pred_item.shape[0]):
            for jj in range(len(eval_k)):
                tmp_hit_num = 0
                tmp_prob = 0.
                for j in range(eval_k[jj]):
                    if pred_item[i][j] in self.gt_list[start_idx+i]:
                        tmp_hit_num += self.gt_list[start_idx+i][pred_item[i][j]]
                    tmp_prob += self.prob_list[start_idx+i][pred_item[i][j]]
                self.hit_num[jj].append(tmp_hit_num)
                p = tmp_hit_num * 1. / eval_k[jj]
                r = tmp_hit_num * 1. / self.gt_num[start_idx+i]
                if p + r > 0:
                    f = 2 * p * r / (p+r)
                else:
                    f = 0.
                avg_p = tmp_prob / eval_k[jj]
                self.prec[jj] += p
                self.recall[jj] += r
                self.f1[jj] += f
                self.prob[jj] += avg_p

    def output(self):
        return_prec = [self.prec[jj]/float(self.data_num) for jj in range(len(eval_k))]
        return_recall = [self.recall[jj]/float(self.data_num) for jj in range(len(eval_k))]
        return_f1 = [self.f1[jj]/float(self.data_num) for jj in range(len(eval_k))]
        return_hit = [sum(self.hit_num[jj])/float(sum(self.gt_num)) for jj in range(len(eval_k))]
        return_prob = [self.prob[jj]/float(self.data_num) for jj in range(len(eval_k))]
        return return_prec, return_recall, return_f1, return_hit, sum(self.gt_num) / float(self.data_num), return_prob

class Data():

    def __init__(self, x, y, yprob, num_label):
        if x is not None:
            self.num_data = x.shape[0]
            self.num_dim = x.shape[1]
        else:
            self.num_data = 0
            self.num_dim = 0
        self.num_label = num_label

        self.yprob = yprob
        self.y = y 
        self.x = x 

    def save(self, path):
        pickle.dump([self.x,self.y, self.num_data, self.num_dim, self.num_label, self.yprob, self.max_upstream], open(path, 'w'))

    def load(self, path):
        self.x, self.y, self.num_data, self.num_dim, self.num_label, self.yprob, self.max_upstream = pickle.load(open(path, 'r'))

    def reshuffle(self):
        idx = range(self.num_data)
        random.shuffle(idx)
        self.x = self.x[idx]
        self.y = [self.y[i] for i in idx]
        self.yprob = [self.yprob[i] for i in idx]
        self.max_upstream = [self.max_upstream[i] for i in idx]

    def preprocess(self, tree):
        yprob_tree = np.zeros((self.num_data,pow(tree.arity, tree.depth-1)))
        for j in range(self.num_label):
            if tree.id2offset[j]-pow(tree.arity, tree.depth-1)+1 < 0:
                print(tree.depth)
            yprob_tree[:,tree.id2offset[j]-pow(tree.arity, tree.depth-1)+1] = self.yprob[:,j]
        self.yprob = yprob_tree
        # upstream
        self.max_upstream = np.zeros((self.num_data,pow(tree.arity, tree.depth)-1)) # 1 for upstream, 0 for not
        for i in range(tree.depth-2,-1,-1):
            tmp = np.stack([yprob_tree[:,0:pow(tree.arity,i+1):2], yprob_tree[:,1:pow(tree.arity,i+1):2]],-1)
            tmp_idx = np.argmax(tmp, -1)
            tmp_onehot = np.zeros((self.num_data, pow(tree.arity, i),2))
            tmp_onehot[:,:,1] = tmp_idx
            tmp_onehot[:,:,0] = 1 - tmp_onehot[:,:,1]
            yprob_tree = np.max(tmp, -1)
            self.max_upstream[:,pow(tree.arity,i+1)-1:pow(tree.arity,i+2)-1] = np.reshape(tmp_onehot, [self.num_data,-1])
        for i in range(self.num_data):
            tmp = [tree.id2offset[self.y[i][j]] for j in range(self.y[i].shape[0])]
            self.y[i] = np.array(tmp)

    def buildTrainBatch(self, num_depth, start_idx, end_idx, style):
        # x: b_size * num_fea, numpy array
        # y: b_size, list
        label = self.y[start_idx:end_idx]
        b_size = len(label)
        if style == 'plt':
            pos_sets, neg_sets = [set() for i in range(b_size)], [set() for i in range(b_size)]
            tmp_label = label
            for j in range(num_depth):
                pos_tmp = [set(tmp_label[i]) for i in range(b_size)]
                pos_sets = [pos_sets[i].union(set(tmp_label[i])) for i in range(b_size)]
                tmp_label = [(item-1)/2 for item in tmp_label]
                left_cand = [list(item*2+1) for item in tmp_label]
                right_cand = [list(item*2+2) for item in tmp_label]
                neg_sets = [neg_sets[i].union(set(left_cand[i]), set(right_cand[i])) for i in range(b_size)]
            neg_sets = [neg_sets[i].difference(pos_sets[i]) for i in range(b_size)]
            pos_lens = [len(pos_sets[i]) for i in range(b_size)]
            neg_lens = [len(neg_sets[i]) for i in range(b_size)]
            maxlen = max([pos_lens[i]+neg_lens[i] for i in range(b_size)])
            label_idx = np.zeros((b_size, maxlen), dtype=np.int32)
            label_val = np.zeros((b_size, maxlen), dtype=np.float32)
            label_mask = np.zeros((b_size, maxlen), dtype=np.float32)
            for i in range(b_size):
                label_idx[i,:pos_lens[i]] = np.array(list(pos_sets[i]), dtype=np.int32)
                label_idx[i,pos_lens[i]:pos_lens[i]+neg_lens[i]] = np.array(list(neg_sets[i]), dtype=np.int32)
                label_val[i,:pos_lens[i]] = 1.
                label_mask[i,:pos_lens[i]+neg_lens[i]] = 1.
            return label_idx, label_val, label_mask, np.sum(label_mask), sum(pos_lens), sum(neg_lens)
        elif style == 'otm-optest' or style == 'tdm':      
            sp_row = np.concatenate([[i] * len(label[i]) for i in range(len(label))]).astype(int)
            sp_col = np.concatenate([range(len(a)) for a in label]).astype(int)
            sp_val = np.concatenate(label).astype(int)
            return_label = []
            for j in range(num_depth):
                tmp_mat = sp.csr_matrix((sp_val, (sp_row, sp_col)), dtype=np.int32)
                tmp_idx = np.transpose(np.stack([sp_row,sp_col]), [1,0])
                return_label.append((tmp_idx, sp_val, tmp_mat.shape))               
                sp_val = (sp_val-1)/2
            return_label = return_label[::-1]
            return return_label # should be return depth
        elif style == 'otm' or style == 'otm-bs':
            # assume all value in the leaf is 1., for recommendation task this will not hold
            return_cand, return_cand_mask, return_cand_upstream = [], [], []
            tmp_label = label
            for j in range(num_depth):
                tmp_pos = [np.unique(item) for item in tmp_label]
                tmp_label = [(item-1)/2 for item in tmp_pos]
                tmp_neigh = [tmp_label[i]*4+3-tmp_pos[i] for i in range(b_size)]
                pos_lens = [len(item) for item in tmp_pos]
                maxlen = max(pos_lens)
                cand = np.zeros((b_size, maxlen, 2), dtype=np.int32)
                cand_mask = np.zeros((b_size, maxlen), dtype=np.float32)
                cand_upstream = np.zeros((b_size, maxlen), dtype=np.int32)
                for i in range(b_size):
                    cand[i,:pos_lens[i],:] = np.stack([tmp_pos[i], tmp_neigh[i]], axis=-1)
                    cand_mask[i,:pos_lens[i]] = 1.
                    tmp_upstream = np.cumsum(tmp_label[i][1:]-tmp_label[i][:-1]>0)
                    cand_upstream[i,1:pos_lens[i]] = tmp_upstream
                    cand_upstream[i,pos_lens[i]:] = cand_upstream[i,pos_lens[i]-1] # to guarantee that map_fn generate tensor 
                return_cand.append(cand)
                return_cand_mask.append(cand_mask)
                return_cand_upstream.append(cand_upstream)
            return return_cand[::-1], return_cand_mask[::-1], return_cand_upstream[::-1]


def build_random_tree(num_label, arity, save_path):
    tree_depth = int(math.log(num_label-1, arity)) + 1
    print('tree depth is %d' % tree_depth)
    rand_idx = np.arange(pow(arity,tree_depth)-1, pow(arity,tree_depth+1)-1)
    np.random.shuffle(rand_idx)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path+'tree.dat', 'w') as file:
        for i in range(num_label):
            file.write('%d:%d\n' % (i, rand_idx[i]))

class Tree():

    def __init__(self):
        self.id2offset = {}
        self.depth = 0
        self.arity = 2
        self.startSampleDepth = 1

    def load(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.split(':')
                self.id2offset[int(item[0])] = int(item[1])
                if int(math.log(int(item[1])+1, self.arity))+1 > self.depth:
                    self.depth = int(math.log(int(item[1])+1, self.arity))+1

class Synthetic():

    def __init__(self, lr, num_id, num_depth, num_dim, Ktrain=50, Ktest=50, style='plt'):

        self.num_depth=num_depth
        self.x_tf = tf.placeholder(dtype=FLOATX, shape=(None, num_dim), name='x_tf') # b_size * num_dim
        self.x_emb = self.x_tf        
        self.b_size = tf.shape(self.x_tf)[0]

        # placeholder for bs
        self.gt = []
        for i in range(num_depth):
            self.gt.append(tf.sparse_placeholder(dtype=tf.int32, name='gt_%d' % i)) # num_depth * (len(gt_i) for i in b_size))
        
        # placeholder for optest
        self.cand, self.cand_mask, self.cand_up = [], [], []
        for i in range(num_depth):
            self.cand.append(tf.placeholder(dtype=tf.int32, name='cand_%d'% i))
            self.cand_mask.append(tf.placeholder(dtype=FLOATX, name='cand_mask_%d' % i))
            self.cand_up.append(tf.placeholder(dtype=tf.int32, name='cand_up_%d'% i))

        # placeholder for negative sampling
        self.label_idx = tf.placeholder(dtype=tf.int32, name='label_idx')
        self.label_val = tf.placeholder(dtype=FLOATX, name='label_val')
        self.label_mask = tf.placeholder(dtype=FLOATX, name='label_mask')

        # linear model + embeddings, 0 for pad
        self.weight = tf.get_variable(name='weight', shape=(num_id, num_dim), dtype=FLOATX, initializer=tf.random_normal_initializer())

        if style == 'otm-optest':
            self.build_train_otm_without_optest(Ktrain)
        elif style == 'otm':
            self.build_train_otm(Ktrain)
        elif style == 'plt':
            self.build_train_plt()
        elif style == 'tdm':
            self.build_train_tdm(Ktrain)
        elif style == 'otm-bs':
            self.build_train_otm_without_bs(Ktrain)
        else:
            print('Not Implemented!')

        if style == 'plt':
            self.beam_layer = self.build_prod_inference(Ktest)
        else:
            self.beam_layer = self.build_inference(Ktest)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def build_prod_inference(self, K):
        b_size = self.b_size
        beam_layer = [tf.zeros([b_size,1], dtype=tf.int32)]
        score_layer = [tf.ones([b_size,1], dtype=FLOATX)]
        for i in range(self.num_depth):
            next_beam = tf.concat([beam_layer[-1]*2+1, beam_layer[-1]*2+2], axis=-1)
            local_weight = tf.nn.embedding_lookup(self.weight, next_beam) # b_size * num_samples * emb_dim
            clean_output = tf.reduce_sum(tf.expand_dims(self.x_emb,1)*local_weight,-1)
            prob_score = tf.nn.sigmoid(clean_output)*tf.concat([score_layer[-1], score_layer[-1]], axis=-1)     
            min_num = tf.minimum(K, pow(2,i+1))
            prob_topk, idx_topk = tf.nn.top_k(prob_score, k=min_num)
            row_idx = tf.tile(tf.expand_dims(tf.range(b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            beam_layer.append(tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num])) # b_size * K
            score_layer.append(tf.reshape(tf.gather_nd(prob_score, all_idx), [-1, min_num]))
        return beam_layer

    def build_inference(self, K):
        b_size = self.b_size
        beam_layer = [tf.zeros([b_size,1], dtype=tf.int32)]
        for i in range(self.num_depth):
            next_beam = tf.concat([beam_layer[-1]*2+1, beam_layer[-1]*2+2], axis=-1)
            local_weight = tf.nn.embedding_lookup(self.weight, next_beam) # b_size * num_samples * emb_dim
            clean_output = tf.reduce_sum(tf.expand_dims(self.x_emb,1)*local_weight,-1)
            min_num = tf.minimum(K, pow(2,i+1))
            prob_topk, idx_topk = tf.nn.top_k(clean_output, k=min_num)
            row_idx = tf.tile(tf.expand_dims(tf.range(b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            beam_layer.append(tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num])) # b_size * K
        return beam_layer

    def build_train_tdm(self, K=10):
        # training with all nodes
        b_size = self.b_size
        loss = []
        beam_layer = [tf.zeros([b_size,1], dtype=tf.int32)]
        for i in range(self.num_depth):
            next_beam = tf.concat([beam_layer[-1]*2+1, beam_layer[-1]*2+2], axis=-1)
            gumbel_noise = -tf.log(-tf.log(tf.random_uniform(shape=tf.shape(next_beam))))
            min_num = tf.minimum(2*K, pow(2,i+1))
            prob_topk, idx_topk = tf.nn.top_k(gumbel_noise, k=min_num) # should it be clean output or output?
            row_idx = tf.tile(tf.expand_dims(tf.range(b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            reorder_beam = tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num]) # b_size * K      
            
            beam_pos = tf.sets.set_intersection(reorder_beam, self.gt[i])
            beam_neg = tf.sets.set_difference(reorder_beam, self.gt[i])
            label_pos = tf.SparseTensor(beam_pos.indices, tf.ones_like(beam_pos.values, dtype=FLOATX), beam_pos.dense_shape)
            label_neg = tf.SparseTensor(beam_neg.indices, tf.zeros_like(beam_neg.values, dtype=FLOATX), beam_neg.dense_shape)
            rebuild_beam = tf.reshape(tf.sparse_concat(sp_inputs=[beam_pos, beam_neg], axis=-1).values, [-1,tf.shape(reorder_beam)[1]])
            rebuild_label = tf.reshape(tf.sparse_concat(sp_inputs=[label_pos, label_neg], axis=-1).values, [-1,tf.shape(reorder_beam)[1]])
                
            train_weight = tf.nn.embedding_lookup(self.weight, rebuild_beam) # b_size * num_samples * emb_dim
            train_output = tf.reduce_sum(tf.expand_dims(self.x_emb,1)*train_weight,-1)

            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=rebuild_label, logits=train_output))
            beam_layer.append(next_beam)
            
        self.loss = tf.concat(loss, axis=-1)

    def build_train_otm_without_optest(self, K=10):
        # training with all nodes
        b_size = self.b_size
        loss = []
        beam_layer = [tf.zeros([b_size,1], dtype=tf.int32)]
        for i in range(self.num_depth):
            next_beam = tf.concat([beam_layer[-1]*2+1, beam_layer[-1]*2+2], axis=-1)
            item_size = tf.minimum(2*K, tf.shape(next_beam)[1])

            beam_pos = tf.sets.set_intersection(next_beam, self.gt[i])
            beam_neg = tf.sets.set_difference(next_beam, self.gt[i])
            label_pos = tf.SparseTensor(beam_pos.indices, tf.ones_like(beam_pos.values, dtype=FLOATX), beam_pos.dense_shape)
            label_neg = tf.SparseTensor(beam_neg.indices, tf.zeros_like(beam_neg.values, dtype=FLOATX), beam_neg.dense_shape)
            rebuild_beam = tf.reshape(tf.sparse_concat(sp_inputs=[beam_pos, beam_neg], axis=-1).values, [-1,tf.shape(next_beam)[1]])
            rebuild_label = tf.reshape(tf.sparse_concat(sp_inputs=[label_pos, label_neg], axis=-1).values, [-1,tf.shape(next_beam)[1]])
                
            train_weight = tf.nn.embedding_lookup(self.weight, rebuild_beam) # b_size * num_samples * emb_dim
            train_output = tf.reduce_sum(tf.expand_dims(self.x_emb,1)*train_weight,-1)
        
            min_num = tf.minimum(K, pow(2,i+1))
            prob_topk, idx_topk = tf.nn.top_k(train_output, k=item_size)
            row_idx = tf.tile(tf.expand_dims(tf.range(b_size), 1), (1, item_size))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            reorder_beam = tf.reshape(tf.gather_nd(rebuild_beam, all_idx), [-1, item_size]) # b_size * K      
            beam_layer.append(reorder_beam[:,:K])
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=rebuild_label, logits=train_output))
            
        self.loss = tf.concat(loss, axis=-1)

    def build_train_otm(self, K=10):
        b_size = self.b_size
        loss = []
        beam_layer = [tf.zeros([b_size,1], dtype=tf.int32)]
        dense_gt = [tf.sparse_tensor_to_dense(self.gt[-1])]
        upstream_score = [self.cand_mask[-1]]
        for_loop = tf.range(b_size)

        # bottom-up to compute label
        for i in range(self.num_depth-2,-1,-1):
            local_weight = tf.nn.embedding_lookup(self.weight, self.cand[i+1]) # b_size * maxlen *2 * emb
            output = tf.reduce_sum(self.x_emb[:, tf.newaxis, tf.newaxis, :]*local_weight,-1) # b_size*maxlen(i)*2
            up_score = self.cand_mask[i+1]*tf.one_hot(tf.argmax(output, -1),depth=2,axis=-1)[:,:,0]*upstream_score[-1] # b_size*maxlen(i)
            maxlen = tf.shape(self.cand[i])[1]-1
            upstream_score.append(tf.map_fn(lambda j: tf.concat([tf.segment_sum(up_score[j], \
                self.cand_up[i+1][j]),tf.zeros((maxlen-self.cand_up[i+1][j,-1],))],0), for_loop, dtype=FLOATX)) # b_size*maxlen(i-1)
            
        upstream_score = upstream_score[::-1]
        self.upstream_score = upstream_score
        # bs train
        self.savelabel = []
        for i in range(self.num_depth):
            next_beam = tf.concat([beam_layer[-1]*2+1, beam_layer[-1]*2+2], axis=-1) # b_size * 2min_num            
            match_mask = tf.cast(tf.equal(next_beam[:,tf.newaxis,:],self.cand[i][:,:,0][:,:,tf.newaxis]),FLOATX) # b_size*maxlen*2min_num
            train_label = tf.stop_gradient(tf.clip_by_value(tf.squeeze(tf.matmul(upstream_score[i][:,tf.newaxis,:],match_mask),axis=1), 0.0,1.0)) # b_size*2min_num
            
            local_weight = tf.nn.embedding_lookup(self.weight, next_beam) # b_size * num_samples * emb_dim
            output = tf.reduce_sum(tf.expand_dims(self.x_emb,1)*local_weight,-1)
            
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label, logits=output))
            
            min_num = tf.minimum(K, pow(2,i+1))
            prob_topk, idx_topk = tf.nn.top_k(output, k=min_num) # should it be clean output or output?
            row_idx = tf.tile(tf.expand_dims(tf.range(b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            beam_layer.append(tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num])) # b_size * K

        self.loss = tf.concat(loss, axis=-1)        

    def build_train_otm_without_bs(self, K=10):
        b_size = self.b_size
        loss = []
        beam_layer = [tf.zeros([b_size,1], dtype=tf.int32)]
        dense_gt = [tf.sparse_tensor_to_dense(self.gt[-1])]
        upstream_score = [self.cand_mask[-1]]
        for_loop = tf.range(b_size)

        # bottom-up to compute label
        for i in range(self.num_depth-2,-1,-1):
            local_weight = tf.nn.embedding_lookup(self.weight, self.cand[i+1]) # b_size * maxlen *2 * emb
            output = tf.reduce_sum(self.x_emb[:, tf.newaxis, tf.newaxis, :]*local_weight,-1) # b_size*maxlen(i)*2
            up_score = self.cand_mask[i+1]*tf.one_hot(tf.argmax(output, -1),depth=2,axis=-1)[:,:,0]*upstream_score[-1] # b_size*maxlen(i)
            maxlen = tf.shape(self.cand[i])[1]-1
            upstream_score.append(tf.map_fn(lambda j: tf.concat([tf.segment_sum(up_score[j], \
                self.cand_up[i+1][j]),tf.zeros((maxlen-self.cand_up[i+1][j,-1],))],0), for_loop, dtype=FLOATX)) # b_size*maxlen(i-1)
            
        upstream_score = upstream_score[::-1]
        self.upstream_score = upstream_score
        # bs train
        self.savelabel = []
        for i in range(self.num_depth):
            next_beam = tf.concat([beam_layer[-1]*2+1, beam_layer[-1]*2+2], axis=-1) # b_size * 2min_num
            gumbel_noise = -tf.log(-tf.log(tf.random_uniform(shape=tf.shape(next_beam))))
            min_num = tf.minimum(2*K, pow(2,i+1))
            prob_topk, idx_topk = tf.nn.top_k(gumbel_noise, k=min_num) # should it be clean output or output?
            row_idx = tf.tile(tf.expand_dims(tf.range(b_size), 1), (1, min_num))
            all_idx = tf.concat((tf.reshape(row_idx, [-1,1]), tf.reshape(idx_topk, [-1,1])), axis=-1)
            reorder_beam = tf.reshape(tf.gather_nd(next_beam, all_idx), [-1, min_num]) # b_size * K      
            
            match_mask = tf.cast(tf.equal(reorder_beam[:,tf.newaxis,:],self.cand[i][:,:,0][:,:,tf.newaxis]),FLOATX) # b_size*maxlen*2min_num
            train_label = tf.stop_gradient(tf.clip_by_value(tf.squeeze(tf.matmul(upstream_score[i][:,tf.newaxis,:],match_mask),axis=1), 0.0,1.0)) # b_size*2min_num
            
            local_weight = tf.nn.embedding_lookup(self.weight, reorder_beam) # b_size * num_samples * emb_dim
            output = tf.reduce_sum(tf.expand_dims(self.x_emb,1)*local_weight,-1)
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label, logits=output))
            
            beam_layer.append(next_beam) 

        self.loss = tf.concat(loss, axis=-1)      

    def build_train_plt(self):
        local_weight = tf.nn.embedding_lookup(self.weight, self.label_idx) # b_size * num_samples * emb_dim
        output = tf.reduce_sum(tf.expand_dims(self.x_emb,1)*local_weight,-1)-self.bias
        self.loss = self.label_mask*(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_val, logits=output))

def buildData(args):
    num_label = args.num_label
    num_fea = args.num_fea
    num_train = args.num_train
    num_test = args.num_test

    weight = np.random.randn(num_fea, num_label)
    train_x = np.random.randn(num_train,num_fea)
    y_prob = 1./(1+np.exp(-np.matmul(train_x,weight)+args.bias)) # num_train * num_label
    train_x = np.concatenate([train_x, np.ones([num_train,1])],-1) # num_train * (num_fea+1)
    train_y = np.random.rand(num_train, num_label)<y_prob # num_train * num_label
    train_y = [np.arange(num_label)[train_y[i]] for i in range(num_train)]
    trainData = Data(train_x, train_y, y_prob, num_label)
    
    test_x = np.random.randn(num_test,num_fea)
    y_prob = 1./(1+np.exp(-np.matmul(test_x,weight)+args.bias)) # num_train * num_label
    test_x = np.concatenate([test_x, np.ones([num_test,1])],-1) # num_test * (num_fea+1)
    test_y = np.random.rand(num_test, num_label)<y_prob # num_train * num_label
    test_y = [np.arange(num_label)[test_y[i]] for i in range(num_test)]
    testData = Data(test_x, test_y, y_prob, num_label)

    return trainData, testData, weight

def gt_result(x, weight, tree):
    score = np.matmul(x, weight)
    idx = np.argsort(score)[:,::-1]
    beam = []
    for i in range(x.shape[0]):
        beam.append(np.array([tree.id2offset[idx[i][j]] for j in range(score.shape[1])]))
    return [np.array(beam)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='plt') 
    parser.add_argument('--num_label', type=int, default=1000)
    parser.add_argument('--num_train', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=1000)
    parser.add_argument('--num_fea', type=int, default=10)
    parser.add_argument('--arity', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--Ktrain', type=int, default=50)
    parser.add_argument('--Ktest', type=int, default=50)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--loadpath', type=str, default='./result/synthetic/')
    parser.add_argument('--testper', type=int, default=10)
    parser.add_argument('--regenerate', type=str, default='False')
    parser.add_argument('--bias', type=float, default=1.)
    args = parser.parse_args()
    print(args)
    eval_k = [1,5,10,20,50]

    loadPath = args.loadpath
    if not os.path.exists(loadPath):
        os.makedirs(loadPath)
    savePath = loadPath
    logfile = open(savePath+args.style+'_log.txt','w')
    for k,v in sorted(vars(args).items()): 
        logfile.write("{0}: {1}\n".format(k,v))

    tree = Tree()
    if not os.path.exists(loadPath+'tree.dat'):
        build_random_tree(args.num_label, args.arity, loadPath)
    tree.load(loadPath+'tree.dat')

    if not os.path.exists(loadPath+'trainData.dat') or args.regenerate=='True':
        print('regenerate data!')
        trainData, testData, gt_weight = buildData(args)
        trainData.preprocess(tree)
        testData.preprocess(tree)
        trainData.save(loadPath+'trainData.dat')
        testData.save(loadPath+'testData.dat')
        np.save(loadPath+'weight', gt_weight)
    else:
        print('load existing data')
        trainData = Data(x=None,y=None,yprob=None,num_label=args.num_label)
        testData = Data(x=None,y=None,yprob=None,num_label=args.num_label)
        trainData.load(loadPath+'trainData.dat')
        testData.load(loadPath+'testData.dat')
        gt_weight = np.load(loadPath+'weight.npy')

    print('data and tree finish')

    model = Synthetic(lr=args.lr, num_id=pow(2,tree.depth), num_depth=tree.depth-tree.startSampleDepth, \
        num_dim=trainData.num_dim, style=args.style, Ktest=args.Ktest, Ktrain=args.Ktrain)
    evaluator = Evaluator(tree, testData.yprob, testData.y, 0, eval_k)
    saver = tf.train.Saver()

    numEpoch = args.num_epoch
    batchSize = args.batch_size
    numBatch = trainData.num_data / batchSize+1
    numTestBatch = testData.num_data / batchSize
    num_depth = tree.depth-tree.startSampleDepth
    tot_batch = 0

    evaluator.reset()
    for j in range(numTestBatch):
        beamLayer = gt_result(testData.x[j*batchSize:(j+1)*batchSize,:-1], gt_weight, tree)
        evaluator.evaluate(beamLayer[-1], j*batchSize)
        
    gt_p, gt_r, gt_f, gt_gr, gt_gt, gt_prob = evaluator.output()
    for jj in range(len(eval_k)):
        print('Ground Truth Weight for Top %d : Precision: %f, Recall: %f, Macro F1: %f, Global Recall: %f, GT Prob: %f, Average GT Num: %f' % \
            (eval_k[jj], gt_p[jj], gt_r[jj], gt_f[jj], gt_gr[jj], gt_prob[jj], gt_gt))
        logfile.write('Ground Truth Weight for Top %d : Precision: %f, Recall: %f, Macro F1: %f, Global Recall: %f, GT Prob: %f, Average GT Num: %f\n' % \
            (eval_k[jj], gt_p[jj], gt_r[jj], gt_f[jj], gt_gr[jj], gt_prob[jj], gt_gt))

    with tf.Session() as sess:
        totSampleNum, totPosNum, totNegNum = 0, 0, 0
        sess.run(tf.global_variables_initializer())
        for idx_epoch in range(numEpoch):
            trainData.reshuffle()
            for idx_batch in range(numBatch):
        x        tot_batch += 1
                start_idx = idx_batch * batchSize
                if start_idx >= trainData.num_data:
                    continue
                batch_x = trainData.x[start_idx:start_idx+batchSize]
                feed_pairs = {model.x_tf:batch_x}
                if args.style == 'otm-optest' or args.style == 'tdm':
                    batchLabel = trainData.buildTrainBatch(num_depth, start_idx, start_idx+batchSize, args.style)
                    for i in range(len(batchLabel)):
                        feed_pairs[model.gt[i]] = batchLabel[i]
                elif args.style == 'plt':
                    batchLabelIdx, batchLabelVal, batchLabelMask, batchSampleNum, batchPosNum, batchNegNum = trainData.buildTrainBatch(num_depth, start_idx, start_idx+batchSize, args.style)
                    feed_pairs[model.label_idx] = batchLabelIdx
                    feed_pairs[model.label_val] = batchLabelVal
                    feed_pairs[model.label_mask] = batchLabelMask
                    totSampleNum += batchSampleNum
                    totPosNum += batchPosNum
                    totNegNum += batchNegNum
                elif args.style == 'otm' or args.style == 'otm-bs':
                    batchCand, batchCandMask, batchCandUp = trainData.buildTrainBatch(num_depth, start_idx, start_idx+batchSize, args.style)
                    for j in range(len(model.cand)):
                        feed_pairs[model.cand[j]] = batchCand[j]
                        feed_pairs[model.cand_mask[j]] = batchCandMask[j]
                        feed_pairs[model.cand_up[j]] = batchCandUp[j]
                _, loss_np = sess.run([model.train_op, model.loss], feed_dict=feed_pairs)

                print('In Batch Iteration %d, training loss is %.4f' % (tot_batch, np.mean(loss_np)))
                logfile.write('In Batch Iteration %d, training loss is %.4f\n' % (tot_batch, np.mean(loss_np)))
            
                if (tot_batch % args.testper == 0 or idx_batch==numBatch-1) and tot_batch != 0:
                    evaluator.reset()
                    for j in range(numTestBatch):
                        beamLayer = sess.run(model.beam_layer, feed_dict={model.x_tf:testData.x[j*batchSize:(j+1)*batchSize]})
                        evaluator.evaluate(beamLayer[-1], j*batchSize)
                        
                    p, r, f, gr, gt, prob = evaluator.output()
                    for jj in range(len(eval_k)):

                        print('Top %d : Regret: %f' % (eval_k[jj], gt_prob[jj]-prob[jj]))
                        logfile.write('Top %d : Regret: %f\n' % (eval_k[jj], gt_prob[jj]-prob[jj]))

        saver.save(sess, savePath + args.style + '-model-final')


