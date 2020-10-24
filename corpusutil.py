import json
import json
import torch.nn.functional as F
import torch
import os

def difbeijing():
    full_sets = [set()] * 10
    for i in range(10):
        print(i)
        train_file = f'''./data/all_deduplicate_filtered_29950_train_{i}.json'''
        dev_file = f'''./data/all_deduplicate_filtered_29950_dev_{i}.json'''
        files = [train_file, dev_file]
        sets = [set(), set()]

        for j in range(2):
            with open(files[j], 'r', encoding='utf8') as f:
                print(files[j])
                import json
                data = json.load(f)
                print(len(data))
                sets[j] |= set([t['id'] for t in data])

        full_sets[i] = sets[0] | sets[1]

        if sets[0] & sets[1]:
            print("error")
            exit()
        else:
            print('pass')

    if full_sets[0] == full_sets[3] == full_sets[6] == full_sets[9]:
        print('pass')
    else:
        print('error')


def split_full_ds():
    import json
    ratio = 0.1
    data = []
    files = [f'./data/0_train.json']
    # files = [f'./data/1_train.json']
    # files = [f'./data/0_train.json', f'./data/1_train.json']
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                example = json.loads(line)
                data.append(example)
    print(len(data))

    train_dev_index = int(len(data) * ratio)
    import random
    seed = 1
    random.seed(seed)
    random.shuffle(data)
    output_file = f'''./data/0_train_{seed}_'''
    with open(output_file + 'train.json', 'w', encoding='utf8') as f:
        print(output_file + 'train.json')
        print(len(data[train_dev_index:]))
        json.dump(data[train_dev_index:], f, ensure_ascii=False, indent=4)
    with open(output_file + 'dev.json', 'w', encoding='utf8') as f:
        print(output_file + 'dev.json')
        print(len(data[:train_dev_index]))
        json.dump(data[:train_dev_index], f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    split_full_ds()
