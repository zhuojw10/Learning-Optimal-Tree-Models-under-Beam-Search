import numpy as np
import sys
import os
import random
import pickle

seqLen = 70
minLen = 8

def process_raw_data(read_path='./data/', write_path='./data/', choice='UserBehavior'):
    train_path = read_path + choice + '_train.csv'
    test_path = read_path + choice + '_test.csv'

    for filepath in [train_path, test_path]:
        user_id, item_id, timestamp = [], [], []
        with open(filepath, 'r') as f:
            for line in f:
                arr = line.strip().split(',')
                user_id.append(int(arr[0]))
                item_id.append(int(arr[1]))
                timestamp.append(int(arr[-1]))

        if filepath == train_path:
            print('Training Data Load Finished.')
        else:
            print('Test Data Load Finished')

        user_dict = {}
        user_feature = []
        for i in range(len(user_id)):
            if user_id[i] not in user_dict:
                user_dict[user_id[i]] = len(user_dict)
                user_feature.append([(item_id[i], timestamp[i])])
            else:
                user_feature[user_dict[user_id[i]]].append((item_id[i], timestamp[i]))
        for i in range(len(user_feature)):
            user_feature[i].sort(key=lambda x: x[1])

        # logic for building training data
        if filepath == train_path:
            print('Training Data Stat: Number of User = %d' % len(user_dict))
            random.shuffle(user_feature)
            build_train_data_with_single_target(user_feature, write_path+choice+'_train_single.dat')
            build_train_data_with_multi_target(user_feature, write_path+choice+'_train_multi.dat')
            print('Training Data Output Finished.')
        else:
            print('Test Data Stat: Number of User = %d' % len(user_dict))
            build_test_data(user_feature, write_path+choice+'_test.dat')
            print('Test Data Output Finished.')

def build_train_data_with_single_target(user_feature, write_path):
    with open(write_path, 'w') as f:
        for i in range(len(user_feature)):
            if len(user_feature[i]) < minLen:
                print(num, len(user_feature[i]))
                continue
            arr = [0 for j in range(seqLen-minLen)] + [v[0] for v in user_feature[i]]
            for j in range(len(arr)-seqLen+1):
                data = arr[j:j+seqLen]
                out_str = str(data[0])
                for k in data[1:]:
                    out_str += ',' + str(k)
                f.write(out_str+'\n')

def build_train_data_with_multi_target(user_feature, write_path):
    with open(write_path, 'w') as f:
        for i in range(len(user_feature)):
            if len(user_feature[i]) / 2 + 1 < minLen:
                continue
            mid = len(user_feature[i]) / 2
            left = user_feature[i][:mid][-seqLen+1:]
            right = user_feature[i][mid:]
            arr = [0 for j in range(seqLen-len(left)-1)] + [v[0] for v in left]
            out_str = str(arr[0])
            for k in arr[1:]:
                out_str += ',' + str(k)
            for k in right:
                out_str += ',' + str(k[0])
            f.write(out_str+'\n')

def build_test_data(user_feature, write_path):
    with open(write_path, 'w') as f:
        for i in range(len(user_feature)):
            mid = len(user_feature[i]) / 2
            left = user_feature[i][:mid][-seqLen+1:]
            right = user_feature[i][mid:]
            arr = [0 for j in range(seqLen-len(left)-1)] + [v[0] for v in left]
            out_str = str(arr[0])
            for k in arr[1:]:
                out_str += ',' + str(k)
            for k in right:
                out_str += ',' + str(k[0])
            f.write(out_str+'\n')

if __name__=='__main__':
    choice = 'UserBehavior' # or 'AmazonBooks'
    process_raw_data(choice=choice)
