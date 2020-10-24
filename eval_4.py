import os
import sys
from trained_models import *

# ir_type = 'bm25'


dataset_path = "/home/zxhuang/MultipleChoice/bert/data/retrieval_final/"
data_dir_dic = {
    "0_train_seed1": lambda ds_type, idx: f"data/0_train_1_{ds_type}.json",
    "1_train_seed1": lambda ds_type, idx: f"data/1_train_1_{ds_type}.json",
    "01_train_seed2": lambda ds_type, idx: f"data/train_01_2_{ds_type}.json",
}
# /home/zxhuang/python_workstation/MultipleChoice/Bert-Choice/data/all_deduplicate
pretrain_map = {
    'ERNIE': "../dcmn/coin/c3pretrain/ERNIE",
    'bert-wwm-ext': "../dcmn/coin/c3pretrain/chinese-bert-wwm-ext",
    'roberta_wwm_large_ext': "../dcmn/coin/c3pretrain/chinese_roberta_wwm_large_ext",
    'roberta_wwm_large_ext_pretrained' : "output/Oct20_14-48-27_ubuntu_LR(3e-05)_DS(['0_train_1_train'])_SEQLEN(128)_TOPK(300)_PNUM(3)_EPOCH(6.0)_model(roberta_wwm_large_ext)"
}
pretrain_model = pretrain_map['bert-wwm-ext']
ir_type = '0_train_seed1'
gpu = "0,1,2"
max_seq_len = 200
# max_seq_len = 400
lr = '3e-5'
epoch = 20
topk = 300
p_num = 3

output = 'output/'
data_dir = "data"

import numpy as np
def get_metrics(out, labels):
    out = np.array(out)
    labels = np.array(labels)
    return {
         'accuracy': (out == labels).mean()
    }

def train_one(idx):
    command = f'''
        CUDA_VISIBLE_DEVICES={gpu} python ./run_dcmn_geo_4.py \
        --model_type bert \
        --model_name_or_path "{pretrain_model}" \
        --do_train \
        --do_lower_case \
        --data_dir {data_dir}\
        --train_file "{data_dir_dic[ir_type]('train', idx)}" \
        --dev_file "{data_dir_dic[ir_type]('dev', idx)}" \
        --max_seq_length {max_seq_len} \
        --model_choices {topk} \
        --p_num {p_num} \
        --batch_size 8 \
        --learning_rate {lr} \
        --gradient_accumulation_steps 1 \
        --num_train_epochs {epoch} \
        --output_dir {output} \
        --seed 1 \
        --overwrite_output_dir \
        --logging_steps 100 \
        --evaluate_during_training \
        --fp16
        --overwrite_cache
    '''
    # beijing 40*2
    print(os.system(command))


def train():
    # for idx in range(0, 10, 3):
    for idx in range(0, 1, 1):
        train_one(idx)


def dev_one(test_model, dev_file, p_num, max_seq_len):
    # for ds_type in ['dev', 'test']:

    for ds_type in ['test']:
        command = f'''
        CUDA_VISIBLE_DEVICES={gpu} python ./do_evaluate_4.py \
        --model_dir "{test_model}" \
        --do_test \
        --model_choices {topk} \
        --data_dir {data_dir} \
        --file "{dev_file}" \
        --max_seq_length {max_seq_len} \
        --batch_size 4 \
        --p_num {p_num} \
        --overwrite_cache
        '''
        print(os.system(command))
        # --start_file "{data_dir_dic[ir_type](ds_type, 0)}" \
        # --end_file "{data_dir_dic[ir_type](ds_type, 4)}" \


def get_seq_len(model):
    import re
    result = re.search(r'SEQLEN\(\d\d\d\)', model)
    if result:
        seq_len = result.group()[7:10]
    else:
        seq_len = 400
    return seq_len


def get_p_num(test_model):
    import re
    result = re.search(r'PNUM\(\d\)', test_model)
    if result:
        pnum = result.group()[5]
    else:
        pnum = 2
    return pnum


def get_output_file(test_model, dev_file):
    # data_dir_dic[ir_type]('test', i)
    p_num = get_p_num(test_model)
    output_file = os.path.join("./ensemble_output_dir",
                               test_model.replace('output/', ''),
                               dev_file.split('/')[-1].replace('.json', f'_{p_num}_output.json'))
    return output_file


def run_models(models, dev_file, run=False):
    for test_model in models:
        test_model = "output/" + test_model
        max_seq_len = get_seq_len(test_model)
        p_num = get_p_num(test_model)
        output_file = get_output_file(test_model, dev_file)
        if run or not os.path.exists(output_file):
            dev_one(test_model, dev_file=dev_file, p_num=p_num, max_seq_len=max_seq_len)


def ensemble_models_every(models, dev_file, run=False, ensemble_type='pairwise'):
    import json
    run_models(models, dev_file=dev_file, run=run)
    every_test_acc = []
    outputs = []
    all_average_acc = []
    for idx, test_model in enumerate(models):
        test_model = "output/" + test_model
        output_file = get_output_file(test_model, dev_file)
        with open(output_file, 'r', encoding='utf8') as f:
            data = json.load(f)
            outputs.append(data)

            if len(data) == 55:
                for index, t in enumerate(data):
                    t['source'] = index // 11
            every_data = []
            paper_sources = sorted(set([s['source'] for s in data]))
            for source in paper_sources:
                every_data.append([t['correct'] for t in data if t['source'] == source])

            print([len(t) for t in every_data])
            every_test_acc.append([sum(test) / len(test) for test in every_data])
            all_average_acc.append(sum([s['correct'] for s in data]) / len(data))

    preds_list = []
    label_list = []
    for j in range(len(outputs[0])):
        preds = [output[j]['predict'] for output in outputs]
        if ensemble_type == "pairwise":
            # print(preds)
            preds_logits = [output[j]['logits'] for output in outputs]
            preds = [[len([s for s in scores if s < t]) for t in scores] for scores in preds_logits]
            preds = np.argmax(np.sum(preds, 0))
        elif ensemble_type == 'even_max':
            if len(set(preds)) == len(outputs):
                preds = outputs[np.argmax([max(output[j]['logits']) for output in outputs])][j]['predict']
            else:
                preds = np.argmax(np.bincount(preds))
        elif ensemble_type == "naive":
            preds = np.argmax(np.bincount(preds))
        # preds = np.argmax(np.bincount(preds))
        preds_list.append(preds)
        label_list.append(outputs[0][j]['label'])

    paper_sources = sorted(set([s['source'] for s in outputs[0]]))
    every_ensemble_preds = []
    every_ensemble_labels = []
    all_ensemble_acc = get_metrics(out=preds_list, labels=label_list)['accuracy']
    for source in paper_sources:
        every_data.append([t for t in data if t['source'] == source])
        every_ensemble_preds.append([preds_list[i] for i in range(len(outputs[0])) if outputs[0][i]['source'] == source])
        every_ensemble_labels.append([label_list[i] for i in range(len(outputs[0])) if outputs[0][i]['source'] == source])
    every_ensemble_acc = np.array([get_metrics(out=preds, labels=labels)['accuracy']
                                   for preds, labels in zip(every_ensemble_preds, every_ensemble_labels)])
    every_test_acc = np.array(every_test_acc)

    np.set_printoptions(precision=4)

    for i in range(len(models)):
        print(f'''model_every_acc_{i} = {every_test_acc[i]}''')

    print(f'''model_acc = {every_test_acc.mean(1)}''')
    print(f"model_average_acc = {every_test_acc.mean()}")
    print(f'all_average_acc = {all_average_acc}')
    print(f'all_average_acc_average = {np.mean(all_average_acc)}')
    print(f"ds_acc = {every_test_acc.mean(0)}")
    print(f"every_ds_ensemble_acc = {every_ensemble_acc}")
    print(f"ensemble_average_acc = {every_ensemble_acc.mean()}")
    print(f"all_ensemble_acc = {all_ensemble_acc}")



import json
def print_example(example):
    print("ID", example['q_id'] if "q_id" in example else example['id'])
    print('背景:', example['background'])
    print("题目:", example['question'])
    if not ('logits' in example):
        example['logits'] = [0, 0, 0, 0]
    # for index, logit in zip(list('ABCD'), example['logits']):
    for index in list('ABCD'):
        # print(f'''{index}-({'%.4f' % logit}):\t{example[index if index in example else "choice_" + index]}''')
        print(f'''{index}:\t{example[index if index in example else (index.lower() if index.lower() in example else "choice_" + index)]}''')
        print('*' * 200)
        para_index = 'paragraph_' + index.lower()
        if para_index not in example:
            para_index[-1].upper()
        paras_gold = example[para_index]
        # paras_gold = list(filter(lambda x:x['answerable'], paras_gold))
        for idx, p in enumerate(paras_gold[:5]):
            print(idx, ':', p['paragraph'])

        print('*' * 200)

    print('logtis:')
    for logit in example['logits']:
        print(logit)

    print('答案:' + example['answer'])
    print("预测:" + str(example['predicted_answer'] if 'predicted_answer' in example else example['predict_ans']))


if __name__ == '__main__':
    # import time
    # time.sleep(60 * 60 *1)
    train()
    # ir_type = 'beijing'

    models = roberta_large_seed1
    p_num = 2
    max_seq_len = 400
    all_models = []
    for model in models:
        test_model = "output/" + model
        if os.path.exists(test_model):
            all_models.append(model)
    models = all_models
    run = False
    ensemble_models_every(models, data_dir_dic[ir_type]('test', 0), run=run, ensemble_type='pairwise')

