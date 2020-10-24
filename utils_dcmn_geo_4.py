'''
this is adaptation of original, for every choice with its own context
'''

import json
import logging

logger = logging.getLogger(__name__)


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""

    def __init__(self,
                 swag_id,
                 ques_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 single_prob,
                 label=None):
        self.swag_id = swag_id
        self.ques_id = ques_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3
        ]
        self.single_prob = single_prob
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"ques_id: {self.ques_id}",
            f"context_sentence: {self.context_sentence}",
            f"start_ending: {self.start_ending}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 ques_id,
                 choices_features,
                 single_prob,
                 label
                 ):
        self.example_id = example_id
        self.ques_id = ques_id
        self.single_prob = single_prob
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len,
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label


def judge_single(statement):
    import re
    expression = r'(哪一|哪种|哪个)'
    if re.search(expression, statement):
        return True
    else:
        return False

def file_reader(paths):
    examples = []
    # print(root.tag)
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf8') as f:
            data += json.load(f)
    for idx, instance in enumerate(data):
        question_text = instance['statement']
        label = 0
        for ans in instance['answer']:
            if ans not in list('ABCD'):
                print(instance)
                continue
            label += 2 ** (ord(ans) - ord('A'))

        if not (0 <= label - 1 <= 14):
            print(instance)
            continue

        examples.append(SwagExample(
            swag_id=idx,
            ques_id=instance['id'],
            context_sentence=[''] * 4,
            start_ending=question_text,  # in the swag dataset, the
            # common beginning of each
            # choice is stored in "sent2".
            ending_0=instance['option_list']['A'],
            ending_1=instance['option_list']['B'],
            ending_2=instance['option_list']['C'],
            ending_3=instance['option_list']['D'],
            single_prob= 1 if judge_single(question_text) else 0,
            label=label - 1
        ))

    print(len(examples))
    return examples

def read_swag_examples(input_file):
    # input_df = pd.read_json(input_file)
    # article, question, ct1, ct2, ct3, ct4, y, q_id, ques_id = file_reader(os.path.join(args.data_dir, input_file), mode=mode)
    examples = file_reader(input_file)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    # outputs.
    features = []
    from tqdm import tqdm
    token_nums = []
    # sentence_nums = []
    for example_index, example in tqdm(enumerate(examples)):
        choices_features = []
        # sentence_index = [0] * 100
        for ending_index, (context, ending) in enumerate(zip(example.context_sentence, example.endings)):
            start_ending_tokens = tokenizer.tokenize(example.start_ending)
            context_tokens = tokenizer.tokenize(context)
            ending_tokens = tokenizer.tokenize(ending)
            token_nums.append(len(context_tokens) + len(start_ending_tokens) + len(ending_tokens))

            # ending_tokens = start_ending_tokens + ending_tokens
            _truncate_seq_tuple(context_tokens, start_ending_tokens, ending_tokens, max_seq_length - 3)
            doc_len = len(context_tokens)
            option_len = len(ending_tokens)
            ques_len = len(start_ending_tokens)

            # print(sentence_index)
            # print(''.join(context_tokens))
            # input()

            # print(sentence_index)
            # if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
            #     ques_len = len(ending_tokens) - option_len
            context_tokens_choice = context_tokens + start_ending_tokens
            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert (doc_len + ques_len + option_len) <= max_seq_length
            if (doc_len + ques_len + option_len) > max_seq_length:
                print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
                assert (doc_len + ques_len + option_len) <= max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))

        label = example.label

        features.append(
            InputFeatures(
                example_id=int(example.swag_id),
                ques_id=int(example.ques_id),
                choices_features=choices_features,
                single_prob=example.single_prob,
                label=label
            )
        )
    import numpy as np
    from scipy import stats
    print(f'average_tokens={np.mean(token_nums)}\n'
          f'tokens_std={np.std(token_nums, ddof=1)}\n'
          f'99%={np.percentile(token_nums, 99)}\n'
          f'{max_seq_length}={stats.percentileofscore(token_nums, max_seq_length)}\n'
          # f'average_sentence={np.mean(sentence_nums)}\n'
          # f'sentence_std={np.std(sentence_nums, ddof=1)}\n'
          # f'90%={np.percentile(sentence_nums, 90)}\n')
          )

    return features


def _truncate_seq_tuple_bak(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_c):
            tokens_b.pop(0)
        else:
            tokens_c.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)

        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            # only truncate the beginning of backgroud+question
            tokens_b.pop(0)
            # tokens_b.pop()

        else:
            tokens_c.pop()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    pop_label = True
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
