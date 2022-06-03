# argparse 사용법을 익히는 것을 목표로!
import torch
import random
import logging
import numpy as np
from argparse import ArgumentParser

# https://greeksharifa.github.io/references/2019/02/12/argparse-usage/
# argparse에 대한 간단하고 풍부한 설명이 있으니 참고!
# argument들을 전부 지정하지 않더라도, default값이 있으니 ㄱㅊ

class Arguments():

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self): # parser에다가 argument를 추가해주는 함수인데...
        self.add_argument('--opt_level', type=str, default='O1')
        self.add_argument('--fp16', type=str, default='True')
        self.add_argument('--train', type=str, default='True')
        self.add_argument('--test', type=str, default='True')
        self.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # cuda 사용 가능시, device를 cuda, 그게 아니면,cpu 사용
        # self.add_argument -> 객체 생성 후, 메서드로 사용됨.

    def add_hyper_parameters(self):
        self.add_argument('--patient', type=int, default=10)
        self.add_argument('--dropout', type=int, default=0.1)
        self.add_argument('--max_len', type=int, default=50)
        self.add_argument('--batch_size', type=int, default=256)
        self.add_argument('--epochs', type=int, default=3)
        self.add_argument('--eval_steps', type=int, default=250)
        self.add_argument('--seed', type=int, default=1234)
        self.add_argument('--lr', type=float, default=0.00005)
        self.add_argument('--weight_decay', type=float, default=0.0)
        self.add_argument('--warmup_ratio', type=float, default=0.05)
        self.add_argument('--temperature', type=float, default=0.05)

        # 필요한 하이퍼 파라미터들.

    def add_data_parameters(self):
        self.add_argument('--train_data', type=str, default='train_nli.tsv')
        self.add_argument('--valid_data', type=str, default='valid_sts.tsv')
        self.add_argument('--test_data', type=str, default='test_sts.tsv')
        self.add_argument('--task', type=str, default='NLU')
        self.add_argument('--path_to_data', type=str, default='./data/')
        self.add_argument('--path_to_save', type=str, default='./output/')
        self.add_argument('--path_to_saved_model', type=str, default='./output/')
        self.add_argument('--ckpt', type=str, default='best_checkpoint.pt')

        # 학습이나 모델 운영시 사용하게 될 data들

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:print("argparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:print("\t", key, ":", value, "\n}")
            else:print("\t", key, ":", value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

        # 한번에 여러개의 입력을 받을 수 있게 해주는 것으로 보임.

    def parse(self):
        args = self.parser.parse_args()
        self.print_args(args)

        return args
        # 명령창에서 주어진 인자를 파싱할 수 있도록 해줌


        # 어렴풋이는 argparse의 목적에 대해서 이해했으나, 어디서 어떤 식으로 사용이 되는지는 아직 잘 모르겠다.
        # 계속 진행하면서 파악해보도록 한다.
        # args는 뭐하는 쉐끼냐...

class Setting():

    def set_logger(self):

        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')
            # 오류 발생시, 어떤 형태로 오류를 출력할지 설정해주는 객체

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        _logger.addHandler(stream_handler)
        _logger.setLevel(logging.DEBUG)
        # debug 수준의 낮은 오류는 따로 파일로 만들지 않고, 그냥 로그에 남긴다.
        return _logger

    def set_seed(self, args):

        seed = args.seed

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # seed 설정

    def run(self):

        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_hyper_parameters()
        parser.add_data_parameters()

        args = parser.parse()
        logger = self.set
        _logger()
        self.set_seed(args)

        return args, logger

