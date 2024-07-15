import numpy as np
import math

def evaluate(beam, gt_prob, m):
    gt_val = np.mean(np.sort(gt_prob)[::-1][:m])
    bs_val = 0.
    for idx in beam:
        if idx in tree.node2label:
            bs_val+=gt_prob[tree.node2label[idx]]/m
    return gt_val, bs_val

def beam_search(model, k):
    beam_layer = [np.zeros(1).astype(int)]
    for i in range(1,model.tree.depth):
        min_num = min(pow(arity,i), k)
        level_idx = np.concatenate([beam_layer[-1]*2+1,beam_layer[-1]*2+2])
        #print(level_idx)
        #level_idx = np.arange(pow(arity,i)-1, pow(arity,i+1))
        level_val = model.node_val[level_idx]
        topk_idx = np.argsort(level_val)[::-1][:min_num]
        beam_layer.append(level_idx[topk_idx])
    return beam_layer

def beam_search_with_parent(model, k):
    beam_layer = [np.zeros(1).astype(int)]
    score_layer = [np.ones(1).astype(int)]
    for i in range(1,model.tree.depth):
        min_num = min(pow(arity,i), k)
        level_idx = np.concatenate([beam_layer[-1]*2+1,beam_layer[-1]*2+2])
        parent_val = np.concatenate([score_layer[-1], score_layer[-1]])
        level_val = model.node_val[level_idx]*parent_val
        topk_idx = np.argsort(level_val)[::-1][:min_num]
        beam_layer.append(level_idx[topk_idx])
        score_layer.append(level_val[topk_idx])
    return beam_layer

class Tree():
    def __init__(self):
        self.label2node = {}
        self.node2label = {}
        self.depth = 0
        self.arity = 0
        self.num_node = 0

    def build_tree(self, num_label, arity):
        self.depth = int(math.log(num_label-1, arity)) + 2
        print('tree depth is %d' % self.depth)
        rand_idx = np.arange(pow(arity,self.depth-1)-1, pow(arity,self.depth)-1)
        np.random.shuffle(rand_idx)
        for i in range(num_label):
            self.label2node[i] = rand_idx[i]
            self.node2label[rand_idx[i]] = i
        self.arity = arity
        self.num_node = pow(arity,self.depth)-1

class GroundTruth():
    def __init__(self, gt_prob, tree, up_choice='direst'):
        self.node_val = np.zeros((tree.num_node,), dtype=np.float32)
        self.inv_node_val = np.ones((tree.num_node,), dtype=np.float32)
        self.tree = tree
        # leaf score
        for i in range(len(gt_prob)):
            self.node_val[tree.label2node[i]] = gt_prob[i]
            self.inv_node_val[tree.label2node[i]] = 1-gt_prob[i]
        if up_choice == 'hierest' or up_choice == 'direst':
            for j in range(tree.depth-2,-1,-1):
                for k in range(pow(arity,j)-1,pow(arity,j+1)-1):
                    self.inv_node_val[k] = self.inv_node_val[2*k+1]*self.inv_node_val[2*k+2]
            for j in range(tree.depth-2,-1,-1):
                for k in range(pow(arity,j)-1,pow(arity,j+1)-1):
                    self.node_val[k] = 1-self.inv_node_val[k]
        elif up_choice == 'optest':
            for j in range(tree.depth-2,-1,-1):
                for k in range(pow(arity,j)-1,pow(arity,j+1)-1):
                    self.node_val[k] = max(self.node_val[k*2+1], self.node_val[k*2+2])

class Model():
    def __init__(self, data, gt_prob, tree, choice='margin', tau=0.1):
        self.node_val = np.zeros((tree.num_node,), dtype=np.float32)
        self.node_data = np.zeros((tree.num_node, data.shape[1]), dtype=np.float32)
        self.gt_node_val = np.zeros((tree.num_node,), dtype=np.float32)
        self.tree = tree
        for i in range(len(gt_prob)):
            self.node_val[tree.label2node[i]] = np.mean(data[i,:])
            self.node_data[tree.label2node[i]] = data[i,:]
            self.gt_node_val[tree.label2node[i]] = gt_prob[i]
        for j in range(tree.depth-2,-1,-1):
            for k in range(pow(arity,j)-1,pow(arity,j+1)-1):
                self.gt_node_val[k] = max(self.gt_node_val[k*2+1], self.gt_node_val[k*2+2])
    
        for j in range(tree.depth-2,-1,-1):
            for k in range(pow(arity,j)-1,pow(arity,j+1)-1):
                if choice == 'direst': # 
                    self.node_data[k] = (self.node_data[k*2+1]+self.node_data[k*2+2]>0).astype(float)
                    self.node_val[k] = np.mean(self.node_data[k])
                elif choice == 'optest':
                    self.node_data[k] = self.node_data[2*k+1] if self.node_val[2*k+1]>self.node_val[2*k+2] else self.node_data[2*k+2]
                    self.node_val[k] = np.mean(self.node_data[k])
                elif choice == 'hierest':
                    tmp = self.node_data[k*2+1]+self.node_data[k*2+2]>0
                    self.node_data[k] = (tmp).astype(float)
                    self.node_val[k*2+1] = np.mean(self.node_data[k*2+1][tmp])
                    self.node_val[k*2+2] = np.mean(self.node_data[k*2+2][tmp])
                    if j == 0:
                        self.node_data[k] = 1.


if __name__ == '__main__':
    num_label = 1000
    arity = 2
    k_list = [1,5,10,20,50]
    choice_list = ['direst', 'hierest', 'optest']
    num_data_list = [100, 1000, 10000, -1]
    num_repeat = 100

    km_list = []
    for i in range(len(k_list)):
        for m in k_list[:i+1]:
            km_list.append((k_list[i], m))

    print(km_list)

    result_np = np.zeros((num_repeat, len(choice_list), len(num_data_list), len(km_list)))

    for repeat_num in range(num_repeat):
        gt_prob = np.random.rand(num_label)
        tree = Tree()
        tree.build_tree(num_label, arity)    
        for idx_choice in range(len(choice_list)):
            for idx_num_data in range(len(num_data_list)):
                choice = choice_list[idx_choice]
                num_data = num_data_list[idx_num_data]
                if num_data > 0:
                    train_data = np.random.rand(num_label, num_data)
                    for i in range(num_label):
                        train_data[i,:] = (train_data[i,:]<gt_prob[i]).astype(float)
                    for idx_km in range(len(km_list)):
                        k, m = km_list[idx_km]
                        model = Model(train_data, gt_prob, tree, choice)
                        if choice == 'hierest':
                            beam = beam_search_with_parent(model, k)
                        else:
                            beam = beam_search(model, k)
                        gt_val, bs_val = evaluate(beam[-1][:m], gt_prob, m)
                        result_np[repeat_num, idx_choice, idx_num_data, idx_km] = gt_val - bs_val
                        print('With choice '+choice+', Beam Search Performance with k=%d and m=%d: gt_val: %.4f, bs_val: %.4f' % (k, m, gt_val, bs_val))
                else:
                    for idx_km in range(len(km_list)):
                        k, m = km_list[idx_km]
                        gt_model = GroundTruth(gt_prob, tree, choice)
                        gt_beam = beam_search(gt_model, k)
                        gt_val, bs_val = evaluate(gt_beam[-1][:m], gt_prob, m)
                        result_np[repeat_num, idx_choice, idx_num_data, idx_km] = gt_val - bs_val
                        print('With choice '+choice+', Beam Search Performance with k=%d and m=%d: gt_val: %.4f, bs_val: %.4f' % (k, m, gt_val, bs_val))

    avg_result = np.mean(result_np, axis=0)
    res_result = np.reshape(avg_result, (-1, len(km_list)))
    np.savetxt('./result/toyresult.csv', np.transpose(res_result, [1,0]), delimiter=' & ', fmt='%.03f')
    np.savetxt('./result/full_toyresult.csv', np.transpose(res_result, [1,0]), delimiter=' & ')

