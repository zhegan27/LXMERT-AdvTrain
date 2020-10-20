# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

from lxrt.entry import convert_sents_to_features

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()

                # ============================ Code for adversarial training =============
                if args.adv_training == True:
                    train_features = convert_sents_to_features(sent, self.model.lxrt_encoder.max_seq_length, self.model.lxrt_encoder.tokenizer)
                    input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
                    embeds_init = self.model.lxrt_encoder.model.bert.embeddings.word_embeddings(input_ids)
                    delta = torch.zeros_like(embeds_init)
                    img_delta = torch.zeros_like(feats)

                    # calculate the prob. scores for clean samples
                    gt_logit = self.model(feats, boxes, sent, adv_training = False, 
                        adv_modality = None, adv_delta_txt = None, 
                        adv_delta_img = None)
                    gt_answer_prob = F.softmax(gt_logit, dim=1)
                    gt_answer_logprob = F.log_softmax(gt_logit, dim=1)

                    # the main loop
                    for astep in range(args.adv_steps):
                        # (0) forward
                        if args.adv_modality == ["text"]:
                            delta.requires_grad_()
                            img_delta = torch.zeros_like(img_embeds_init)
                        elif args.adv_modality == ["image"]:
                            img_delta.requires_grad_()
                            delta = torch.zeros_like(embeds_init)
                        else:
                            delta.requires_grad_()
                            img_delta.requires_grad_()

                        logit = self.model(feats, boxes, sent, adv_training = True,
                            adv_modality = args.adv_modality, adv_delta_txt = delta,
                            adv_delta_img = img_delta)

                        # BCE loss
                        bce_loss = self.bce_loss(logit, target) 
                        bce_loss = bce_loss * logit.size(1)

                        # KL loss
                        answer_prob = F.softmax(logit, dim=1)
                        answer_logprob = F.log_softmax(logit, dim=1)
                        kl_loss = F.kl_div(answer_logprob,gt_answer_prob,reduction='none') + \
                                    F.kl_div(gt_answer_logprob,answer_prob,reduction='none')
                        kl_loss = kl_loss.mean() * logit.size(1) 

                        # (1) backward
                        loss = (bce_loss + args.adv_kl_weight * kl_loss) / args.adv_steps

                        loss.backward(retain_graph=True)

                        if astep == args.adv_steps - 1:
                            # further updates on delta
                            break

                        # (2) get gradient on delta
                        if "text" in args.adv_modality:
                            delta_grad = delta.grad.clone().detach().float()
                        if "image" in args.adv_modality:
                            img_delta_grad = img_delta.grad.clone().detach().float()

                        # (3) update and clip for delta
                        if "text" in args.adv_modality:
                            if args.norm_type == "l2":
                                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                delta_step = (args.adv_lr_txt * delta_grad / denorm).to(delta)
                                delta = (delta + delta_step).detach()
                                if args.adv_max_norm > 0:
                                    delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).detach()
                                    exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                                    reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1)
                                    delta = (delta * reweights).detach()
                            elif args.norm_type == "linf":
                                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                delta_step = (args.adv_lr_txt * delta_grad / denorm).to(delta)
                                delta = (delta + delta_step).detach()
                                if args.adv_max_norm > 0:
                                    delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()

                        # (4) update and clip for image delta
                        if "image" in args.adv_modality:
                            if args.norm_type == "l2":
                                denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                img_delta_step = (args.adv_lr_img * img_delta_grad / denorm).to(img_delta)
                                img_delta = (img_delta + img_delta_step).detach()
                                if args.adv_max_norm > 0:
                                    delta_norm = torch.norm(img_delta.view(img_delta.size(0), -1), p=2, dim=1).detach()
                                    exceed_mask = (delta_norm > args.adv_max_norm).to(img_embeds_init)
                                    reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1)
                                    img_delta = (img_delta * reweights).detach()
                            elif args.norm_type == "linf":
                                denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                                denorm = torch.clamp(denorm, min=1e-8)
                                img_delta_step = (args.adv_lr_img * img_delta_grad / denorm).to(img_delta)
                                img_delta = (img_delta + img_delta_step).detach()
                                if args.adv_max_norm > 0:
                                    img_delta = torch.clamp(img_delta, -args.adv_max_norm, args.adv_max_norm).detach()
                else:
                    logit = self.model(feats, boxes, sent, adv_training = False,
                        adv_modality = None, adv_delta_txt = None, adv_delta_img = None)
                    assert logit.dim() == target.dim() == 2
                    if args.mce_loss:
                        max_value, target = target.max(1)
                        loss = self.mce_loss(logit, target) * logit.size(1)
                    else:
                        loss = self.bce_loss(logit, target)
                        loss = loss * logit.size(1)

                    loss.backward()
                    
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent, adv_training = False,
                    adv_modality = None, adv_delta_txt = None, adv_delta_img = None)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)


