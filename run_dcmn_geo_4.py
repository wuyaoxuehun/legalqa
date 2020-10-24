import logging
import os
import random

from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
)

from BertForDCMNGeo_4 import BertForMultipleChoiceWithMatch
from DUMA import DUMA

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, DistilBertConfig, XLMConfig)), ()
)

MODEL_CLASSES = {
    # "bert": (BertConfig, BertForMultipleChoiceWithMatch, BertTokenizer),
    "bert": (BertConfig, DUMA, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from utils_dcmn_geo_4 import convert_examples_to_features, read_swag_examples, select_field

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


#
# import pydevd_pycharm
# # pydevd_pycharm.settrace('114.212.84.202', port=8888, stdoutToServer=True, stderrToServer=True)
# pydevd_pycharm.settrace('172.26.126.18', port=8888, stdoutToServer=True, stderrToServer=True)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_metrics(out, labels):
    out = np.array(out)
    labels = np.array(labels)
    return {
         'accuracy': (out == labels).mean()
    }


def train(args, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size * args.n_gpu)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size * args.n_gpu * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    if args.local_rank in [-1, 0]:
        train_files = [file.split('/')[-1].split('.')[0] for file in args.train_file.split(',')]
        tb_writer = SummaryWriter(flush_secs=10, comment='_LR(%.e)_DS(%s)_SEQLEN(%s)_bs(%s)_PNUM(%s)_EPOCH(%s)_model(%s)'
                                                         % (args.learning_rate,
                                                            train_files,
                                                            args.max_seq_length,
                                                            (args.batch_size, args.gradient_accumulation_steps),
                                                            args.p_num,
                                                            args.num_train_epochs,
                                                            'roberta_wwm_ext' if args.model_name_or_path.split('/')[-1].find('roberta_wwm_ext') != -1 else
                                                            ('bert-wwm-ext' if args.model_name_or_path.split('/')[-1].find('bert-wwm-ext') != -1 else
                                                             ("roberta_wwm_large_ext" if args.model_name_or_path.split('/')[-1].find('roberta_wwm_large_ext') != -1 else
                                                             ("ERNIE" if args.model_name_or_path.split('/')[-1].find('ERNIE') != -1 else "unknown")))))
        try:
            args.log_dir = tb_writer.logdir
            args.output_subdir = os.path.join(args.output_dir, tb_writer.logdir.split('/')[-1])
        except Exception as e:
            logger.info(e)
            args.log_dir = tb_writer.log_dir
            args.output_subdir = os.path.join(args.output_dir, tb_writer.log_dir.split('/')[-1])

    best_metrics = 0
    ori_patience = 15
    patience = ori_patience

    # args.logging_steps = args.logging_steps - args.logging_steps % args.gradient_accumulation_steps
    for epoch in train_iterator:
        updated_best = False
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            try:
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                # batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len, single_prob = batch

                loss, logits = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len, single_prob, label_ids)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
            except Exception as e:
                import shutil
                shutil.rmtree(args.log_dir)
                logger.info('deleted logdir')
                logger.info(e)
                exit()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0) \
                        or ((step + 1 == len(epoch_iterator) - len(epoch_iterator) % args.gradient_accumulation_steps)
                            and epoch == len(train_iterator) - 1):
                    results, _, _ = evaluate(args, model, tokenizer)

                    if args.overwrite_cache:
                        args.overwrite_cache = False
                    tb_writer.add_scalar("loss/eval_loss", results['loss'], global_step)
                    tb_writer.add_scalar('metrics/accuracy', results['accuracy'], global_step)

                    if results['accuracy'] > best_metrics:
                        updated_best = True
                        best_metrics = results['accuracy']
                        save_model(args, model, tokenizer)
                        best_results = {'epoch': epoch, 'global_step': global_step}
                        best_results.update(results)
                        save_best_results(args, best_results)
                    # save_model(args, model, tokenizer, step=global_step)

                    logger.info("Current best Accuracy = %s", best_metrics)
                    tb_writer.add_scalar("train_lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss/train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
        if updated_best:
            patience = ori_patience
        else:
            patience -= 1
            if patience == 0:
                break

    # from utils import write_hparams_and_metrics
    # write_hparams_and_metrics(args, best_results)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def save_model(args, model, tokenizer, step=-1):
    save_dir = args.output_subdir
    if step >= 0:
        save_dir = os.path.join(save_dir, f'''step({step})''')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info("Saving model to %s", save_dir)
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, "training_args.bin"))


def save_best_results(args, results):
    if not os.path.exists(args.output_subdir):
        os.makedirs(args.output_subdir)

    eval_file = os.path.join(args.output_subdir, 'eval_results.txt')
    logger.info("******** Saving best results to %s*********** ", eval_file)
    with open(eval_file, 'w', encoding='utf8') as writer:
        for k, v in results.items():
            logger.info("%s = %s" % (k, v))
            writer.write('%s=%s' % (k, v))


def evaluate(args, model, tokenizer, mode='dev'):
    results = {}
    eval_dataset = load_and_cache_examples(args=args,
                                           file=args.dev_file if mode == 'dev' else args.test_file,
                                           tokenizer=tokenizer,
                                           mode=mode)
    preds, gts, loss = get_model_predict(args, model=model,
                                         dataset=eval_dataset)

    preds_label = np.argmax(preds, axis=1)
    metrics = get_metrics(preds_label, gts)
    results.update(metrics)
    results.update({'loss': loss})

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds.tolist(), gts


def get_model_predict(args, model, dataset):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # test!
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_labels = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len, single_prob = batch
            tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len, single_prob, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_labels = np.append(out_labels, label_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    np.set_printoptions(threshold=np.inf)
    # preds = np.argmax(preds, axis=1)
    return preds, out_labels.tolist(), eval_loss


def load_and_cache_examples(args, file, tokenizer, mode):
    if type(file) != list:
        # file = [file]
        file = file.split(',')
    cached_features_file = os.path.join(
        os.path.join(args.data_dir, 'cache'),
        "cached_{}_{}_{}_{}".format([fi.split('/')[-1].replace('.json', '') for fi in file], mode, args.p_num, str(args.max_seq_length)),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", file)
        examples = read_swag_examples(file)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_seq_length=args.max_seq_length
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    # features = features[:100]
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_doc_len = torch.tensor(select_field(features, 'doc_len'), dtype=torch.long)
    all_ques_len = torch.tensor(select_field(features, 'ques_len'), dtype=torch.long)
    all_option_len = torch.tensor(select_field(features, 'option_len'), dtype=torch.long)
    # all_sentence_index = torch.tensor(select_field(features, 'sentence_index'), dtype=torch.long)
    all_single_prob = torch.tensor([f.single_prob for f in features], dtype=torch.long)

    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                            all_option_len, all_single_prob)
    return dataset


def do_train():
    from config import train_args
    args = train_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = 0 if args.no_cuda else 1
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.device = device

    # Setup logging

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=4,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    logger.info("Training parameters %s", args)
    # Training
    if args.do_train:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            # model_choices=args.model_choices
        )

        # model.to(args.device)
        model.cuda()
        train_dataset = load_and_cache_examples(args,
                                                args.train_file,
                                                tokenizer,
                                                mode='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    do_train()
