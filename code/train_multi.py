from transformers import AutoTokenizer, BertConfig
from fairseq.data import data_utils
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from model.optim import ScheduledOptim, Adam
from tqdm import tqdm
import argparse
import os
from eval import evaluate
from model.contrast_multi import ContrastModel, BertEmbeddings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from sklearn.model_selection import KFold
from accelerate import Accelerator
import utils


class BertDataset(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None):
        self.device = device
        super(BertDataset, self).__init__()
        self.data = data_utils.load_indexed_dataset(
            data_path + '/tok', None, 'mmap'
        )
        self.labels = data_utils.load_indexed_dataset(
            data_path + '/Y', None, 'mmap'
        )
        self.max_token = max_token
        self.pad_idx = pad_idx

    def __getitem__(self, item):
        data = self.data[item][:self.max_token - 2].to(
            self.device)
        labels = self.labels[item].to(self.device)
        return {'data': data, 'label': labels, 'idx': item, }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx


class CommonBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(CommonBertEmbeddings, self).__init__()
        self.embeddings = BertEmbeddings(config)


class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name=None):
        emb_name = emb_name  
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name is param:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name=None):
        emb_name = emb_name   # 设置默认值
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name is param:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='WebOfScience', choices=['WebOfScience', 'nyt', 'rcv1'],
                    help='Dataset.')
parser.add_argument('--batch', type=int, default=4, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=10, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast', default=1, type=int, help='Whether use contrastive model.')
parser.add_argument('--graph', default=1, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.')
parser.add_argument('--multi', default=True, action='store_false',
                    help='Whether the task is multi-label classification.')
parser.add_argument('--lamb', default=1, type=float, help='lambda')
parser.add_argument('--thre', default=0.02, type=float,
                    help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')
parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')
parser.add_argument('--experts', type=int, default=3, help='Number of experts')
parser.add_argument('--ta', default=0.5, type=float,
                    help='If prefix weight ≤ tau , the loss of expert m on the sample will be eliminated.')
parser.add_argument('--eta', default=0.91, type=float,
                    help='eta is a temperature factor that adjusts the sensitivity of prefix weights.')

def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_layer_parameters(layer):
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)

def get_final_output(x, y):
    index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    index.scatter_(1, y.data.view(-1, 1), 1)
    index_float = index.float()
    batch_m = torch.matmul(m_list[None, :],
                           index_float.transpose(0, 1))
    batch_m = batch_m.view((-1, 1))
    x_m = x - batch_m
    return torch.exp(
        torch.where(index, x_m, x))

def to(self, device):
    super().to(device)
    self.m_list = self.m_list.to(device)
    if self.per_cls_weights_enabled is not None:
        self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
    if self.per_cls_weights_enabled_diversity is not None:
        self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)
    return self


def _hook_before_epoch(self, epoch):
    if self.reweight_epoch != -1:
        epoch = epoch
        if epoch > self.reweight_epoch:
            self.per_cls_weights_base = self.per_cls_weights_enabled
            self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
        else:
            self.per_cls_weights_base = None
            self.per_cls_weights_diversity = None


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly = True
    args = parser.parse_args()
    device = args.device
    accelerator = Accelerator()
    print(args)
    if args.wandb:
        import wandb
        wandb.init(config=args, project='htc')
    utils.seed_torch(args.seed)
    args.name = args.data + '-' + args.name
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    data_path = os.path.join('data', args.data)
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)
    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    ta = args.ta
    eta = args.eta
    reweight_temperature = eta
    annealing = 500
    experts = args.experts
    print(experts)
    class_count = {}
    for item in dataset:
        label = item['label'].tolist()
        for i, element in enumerate(label):
            if i not in class_count:
                class_count[i] = {}

            if element not in class_count[i]:
                class_count[i][element] = 1
            else:
                class_count[i][element] += 1

    class_count_list = []
    for class_dict in class_count.values():
        counts = list(class_dict.values())
        class_count_list.append(counts)
    class_count_list = [class_count[i][1] for i in sorted(class_count.keys())]


    reweight_epoch = -2
    reweight_factor = 0.05
    m_list = 1. / np.sqrt(np.sqrt(class_count_list))
    max_m = 0.5
    m_list = m_list * (max_m / np.max(m_list))
    m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
    m_list = m_list.to(device)

    if reweight_epoch != -1:
        idx = 1
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], class_count_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_count_list)
        per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)

    else:
        per_cls_weights_enabled = None
    per_cls_weights_enabled = per_cls_weights_enabled.to(device)
    class_count_list = np.array(class_count_list) / np.sum(class_count_list)

    C = len(class_count_list)

    per_cls_weights = C * class_count_list * reweight_factor + 1 - reweight_factor
    per_cls_weights = per_cls_weights / np.max(per_cls_weights)
    T = (reweight_epoch + annealing) / reweight_factor

    per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).to(
        device)
    per_cls_weights_diversity = per_cls_weights_enabled_diversity

    if reweight_epoch != -1:
        per_cls_weights_base = per_cls_weights_enabled
    else:
        per_cls_weights_base = None


    models = []
    optimizers = []
    savers = []
    config = BertConfig(num_labels=num_class,
                        contrast_loss=args.contrast, graph=args.graph,
                        layer=args.layer, data_path=data_path, multi_label=args.multi,
                        lamb=args.lamb, threshold=args.thre, tau=args.tau)
    lora_config = LoraConfig(
        r=128,
        lora_alpha=4096,
        target_modules=['bert.encoder.layer.11.intermediate.dense','bert.encoder.layer.11.output.dense'],
        lora_dropout=0.05,
        bias="none",
    )

    for i in range(experts):
        model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                              contrast_loss=args.contrast, graph=args.graph,
                                              layer=args.layer, data_path=data_path, multi_label=args.multi,
                                              lamb=args.lamb, threshold=args.thre, tau=args.tau, name=args.name).to(device)
        if i == 0:
            for name, module in model.named_modules():
                print(f"Layer Name: {name}, Layer Type: {module.__class__.__name__}")
            print(f"Total number of parameters in the model: {count_parameters(model)}")
            models.append(model)
            fgm = FGM(model)
        else:
            model = get_peft_model(model, lora_config)
            print(f"Total number of parameters in the model: {count_parameters(model)}")
            models.append(model)
            fgm = FGM(model)
        if args.warmup > 0:
            optimizer = ScheduledOptim(Adam(model.parameters(),
                                            lr=args.lr), args.lr,
                                       n_warmup_steps=args.warmup)
        else:
            optimizer = Adam(model.parameters(),
                             lr=args.lr)
        optimizers.append(optimizer)

        saver = Saver(model, optimizer, None, args)
        savers.append(saver)


    if args.wandb:
        for model in models:
            wandb.watch(model)

    split = torch.load(os.path.join(data_path, 'split.pt'))
    train1 = Subset(dataset, split['train'])
    dev1 = Subset(dataset, split['val'])
    dataset_size1 = len(train1)
    combined_dataset = ConcatDataset([train1, dev1])
    n_splits = 5
    min_length_per_fold = len(combined_dataset) // n_splits
    min_length_per_fold = min_length_per_fold + 1 if len(combined_dataset) % n_splits != 0 else min_length_per_fold

    fold_datasets = []

    split_data_count = 0

    for fold in range(n_splits):
        current_fold_length = min_length_per_fold if fold < len(
            combined_dataset) % (n_splits) else min_length_per_fold - 1
        current_fold_dataset = Subset(combined_dataset, range(split_data_count, split_data_count + current_fold_length))
        fold_datasets.append(current_fold_dataset)
        split_data_count += current_fold_length

    for fold, fold_dataset in enumerate(fold_datasets):
        if fold == 1:
            train_indices = list(set(range(len(combined_dataset))) - set(fold_dataset.indices))
            train_dataset = Subset(combined_dataset, train_indices)
            train = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=dataset.collate_fn)
            dev = DataLoader(fold_dataset, batch_size=args.batch, shuffle=False, collate_fn=dataset.collate_fn)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))
    log_file = open(os.path.join('checkpoints', args.name, 'log.txt'), 'w')
    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break
        for model in models:
            model.train()
        i = 0
        loss = 0
        # Train
        pbar = tqdm(train)
        for data, label, idx in pbar:
            outputs = []
            padding_mask = data != tokenizer.pad_token_id
            for model in models:
                output = model(data, padding_mask, labels=label, return_dict=True, return_pooled_output=True)
                outputs.append(output)
            xis = [None] * len(outputs)
            # evidential
            for i in range(len(outputs)):
                xis[i] = outputs[i]['logits']
            num_classes = outputs[0]['num_labels']
            w = [torch.ones(len(xis[0]), dtype=torch.bool, device=xis[0].device)]
            b0 = None

            for xi in xis:
                alpha = torch.exp(xi) + 1
                S = alpha.sum(dim=1, keepdim=True)
                b = (alpha - 1) / S
                u = num_classes / S.squeeze(-1)

                if b0 is None:
                    C = 0
                else:
                    bb = b0.view(-1, b0.shape[1], 1) @ b.view(-1, 1, b.shape[1])
                    C = bb.sum(dim=[1, 2]) - bb.diagonal(dim1=1, dim2=2).sum(dim=1)
                b0 = b
                w.append(w[-1] * u / (1 - C))
            exp_w = [torch.exp(wi / eta) for wi in w]
            exp_w = exp_w[:-1]
            exp_w_sum = sum(exp_w)
            normalized_list = [x / exp_w_sum for x in exp_w]
            exp_w = normalized_list
            exp_w = [wi.unsqueeze(-1) for wi in exp_w]
            reweighted_outs = []
            for i in range(len(xis)):
                reweighted_outs.append(xis[i] * exp_w[i])
            xi = torch.mean(torch.stack(reweighted_outs), dim=0)
            yi = outputs[0]['labels']
            y = torch.argmax(yi, dim=1)
            l_values = []
            kl_values = []

            for i in range(len(xis)):
                alpha = get_final_output(xis[i], y)
                S = alpha.sum(dim=1, keepdim=True)

                l = F.nll_loss(torch.log(alpha) - torch.log(S), y, weight=per_cls_weights_enabled,
                               reduction="none")
                alpha_tilde = yi + (1 - yi) * (alpha + 1)
                S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
                kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(alpha_tilde.shape[1])) - torch.lgamma(
                    alpha_tilde).sum(dim=1, keepdim=True) \
                     + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1,
                                                                                                       keepdim=True)
                l += epoch / T * kl.squeeze(-1)
                l_values.append(l)
                kl_values.append(kl)
            w = w[:-1]
            wmax = [0] * 9
            batch_num_elements = w[0].numel()
            for i in range(batch_num_elements):
                wmax[i] = max(tensor[i].item() for tensor in w)

            for i in range(len(w)):
                for j in range(batch_num_elements):
                    w[i][j] = w[i][j] / wmax[j]
                w[i] = torch.where(w[i] > ta, True, False)
                if w[i].sum() == 0:
                    l_values[i] = 0
                else:
                    l_values[i] = (w[i] * l_values[i]).sum() / w[i].sum()

            loss /= args.update
            outputloss = 0
            for i in range(len(outputs)):
                mask = w[i].type_as(outputs[i]['loss'])
                masked_loss = outputs[i]['loss'] * mask
                masked_l_values = l_values[i] * mask
                current_loss = masked_loss.sum() + masked_l_values.sum()
                outputloss += current_loss
            accelerator.backward(outputloss)
            loss += outputloss.item()
            i += 1
            if i % args.update == 0:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

                if args.wandb:
                    wandb.log({'train_loss': loss})
                pbar.set_description('loss:{:.4f}'.format(loss))
                i = 0
                loss = 0
            torch.cuda.empty_cache()
        pbar.close()

        for model in models:
            model.eval()
        pbar = tqdm(dev)
        with torch.no_grad():
            truth = []
            pred = []
            for data, label, idx in pbar:
                outputs = []
                padding_mask = data != tokenizer.pad_token_id
                for model in models:
                    output = model(data, padding_mask, labels=label, return_dict=True)
                    outputs.append(output)
                xis = [None] * len(outputs)
                for i in range(len(outputs)):
                    xis[i] = outputs[i]['logits']
                num_classes = outputs[0]['num_labels']
                w = [torch.ones(len(xis[0]), dtype=torch.bool, device=xis[0].device)]
                b0 = None

                for xi in xis:
                    alpha = torch.exp(xi) + 1
                    S = alpha.sum(dim=1, keepdim=True)
                    b = (alpha - 1) / S
                    u = num_classes / S.squeeze(-1)

                    if b0 is None:
                        C = 0
                    else:
                        bb = b0.view(-1, b0.shape[1], 1) @ b.view(-1, 1, b.shape[1])
                        C = bb.sum(dim=[1, 2]) - bb.diagonal(dim1=1, dim2=2).sum(dim=1)
                    b0 = b
                    w.append(w[-1] * u / (1 - C))
                exp_w = [torch.exp(wi / eta) for wi in w]
                exp_w = exp_w[:-1]
                exp_w_sum = sum(exp_w)
                normalized_list = [x / exp_w_sum for x in exp_w]
                exp_w = normalized_list
                exp_w = [wi.unsqueeze(-1) for wi in exp_w]
                reweighted_outs = []
                for i in range(len(xis)):
                    reweighted_outs.append(xis[i] * exp_w[i])
                xi = torch.mean(torch.stack(reweighted_outs), dim=0)
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in xi:
                    pred.append(torch.sigmoid(l).tolist())
        pbar.close()
        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        print('macro', macro_f1, 'micro', micro_f1, file=log_file)
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1, 'best_macro': best_score_macro,
                       'best_micro': best_score_micro})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            for i, saver in enumerate(savers, start=1):
                extra = f'_macro{i}'
                checkpoint_path = os.path.join('checkpoints', args.name, f'checkpoint_best{extra}.pt')
                saver(macro_f1, best_score_macro, checkpoint_path)

            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            for i, saver in enumerate(savers, start=1):
                extra = f'_micro{i}'
                checkpoint_path = os.path.join('checkpoints', args.name, f'checkpoint_best{extra}.pt')
                saver(micro_f1, best_score_micro, checkpoint_path)

            early_stop_count = 0
    log_file.close()
