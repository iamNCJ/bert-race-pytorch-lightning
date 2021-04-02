import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace
from transformers import BertConfig, BertTokenizer, BertForMultipleChoice, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

import os
from tqdm import tqdm
import json
import logging
import numpy as np

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''
    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = 4):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)



class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def read_race_examples(filename):
    examples = []
    with open(filename, 'r', encoding='utf-8') as fpr:
        data_raw = json.load(fpr)
        article = data_raw['article']
        ## for each qn
        for i in range(len(data_raw['questions'])):
            question = data_raw['questions'][i]
            options = data_raw['options'][i]
            examples.append(
                RaceExample(
                    race_id = filename+'-'+str(i),
                    context_sentence = article,
                    start_ending = question,

                    ending_0 = options[0],
                    ending_1 = options[1],
                    ending_2 = options[2],
                    ending_3 = options[3],
                    # label = truth
                ))

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option
    #
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    max_option_len = 0
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"

            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

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

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        features.append(
            InputFeatures(
                example_id = example.race_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # Note: only truncate sequence A
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop()
        # if len(tokens_a) > len(tokens_b):
        #     tokens_a.pop()
        # else:
        #     tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./model/bert-large-uncased/vocab.txt', use_fast=True, do_lower_case=True)

    # Prepare model
    bert_model = './model/bert-large-uncased' # !! Note that this is NOT the real path
    config = BertConfig.from_pretrained(bert_model+'/bert_config.json', num_choices=4)
    model = BertForMultipleChoice.from_pretrained(bert_model, config=config)
    model.to(device)

    ## test
    data_dir = './test_wow'
    output_dir = './output_predict'
    has_ans = False
    max_seq_length = 512
    filenames = os.listdir(data_dir)
    model.eval()
    eval_iter = tqdm(filenames, disable=False)
    eval_answers = {}
    eval_accuracy = 0
    nb_eval_examples = 0
    for fid, filename in enumerate(eval_iter):
        file_path = os.path.join(data_dir, filename)
        eval_examples = read_race_examples(file_path)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, max_seq_length, True)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
        eval_answer = []
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            eval_answer.extend(chr(np.argmax(logits.logits.detach().cpu().numpy(), axis=1)[0] + ord("A")))
            if has_ans:
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)
                # print(label_ids, np.argmax(logits, axis=1), tmp_eval_accuracy)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += input_ids.size(0)

        eval_answers[filename] = eval_answer
        # final_eval_accuracy = eval_accuracy / nb_eval_examples
        # logger.info("eval accuracy: {}".format(final_eval_accuracy))
    result = json.dumps(eval_answers, indent=2)

    output_eval_file = os.path.join(output_dir, "answers.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_eval_file, "w") as f:
        f.write("{\n")
        for (key, value) in eval_answers.items():
            f.write('  "{}":{},\n'.format(key, str(value).replace("'", '"')))
        f.write("}\n")
    print(filenames)
