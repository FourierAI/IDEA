import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of IDEA-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_IDEA(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    cache_keys = (im_cache_keys+text_cache_keys)/2
    # Zero-shot CLIP
    zero_logits = 100. * val_features @ clip_weights
    acc = cls_acc(zero_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # IDEA-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    idea_logits = zero_logits + few_logits * alpha
    acc = cls_acc(idea_logits, val_labels)
    print("**** IDEA-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_theta, best_beta, best_alpha = search_hp_2(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    zero_logits = 100. * test_features @ clip_weights
    acc = cls_acc(zero_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # IDEA-Adapter    
    affinity = best_theta* test_features @ text_cache_keys + (1-best_theta) * test_features @ im_cache_keys
    few_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    idea_logits = zero_logits + few_logits * best_alpha
    acc = cls_acc(idea_logits, test_labels)
    print("**** IDEA-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_TIDEA(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):

    adapter = nn.Linear(im_cache_keys.shape[0], im_cache_keys.shape[0], bias=True).to(clip_model.dtype).cuda()
    adapter2 = nn.Linear(im_cache_keys.shape[0], im_cache_keys.shape[1], bias=True).to(clip_model.dtype).cuda()
    
    optimizer = torch.optim.AdamW(
        [{"params":adapter.parameters()},
         {"params":adapter2.parameters()}], 
                                  lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        adapter2.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target, dess) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            image_features_text = adapter(image_features)
            affinity2 = adapter2(image_features)

            affinity = (image_features_text @ text_cache_keys + image_features @ im_cache_keys + image_features @ text_cache_keys)/3
            affinity += affinity2

            few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            zero_logits = 100. * image_features @ clip_weights
            TIDEA_logits = zero_logits + few_logits * alpha

            loss = F.cross_entropy(TIDEA_logits, target)

            acc = cls_acc(TIDEA_logits, target)
            correct_samples += acc / 100 * len(TIDEA_logits)
            all_samples += len(TIDEA_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()
        adapter2.eval()
        test_features_text = adapter(test_features)
        affinity2 = adapter2(test_features)

        affinity = (test_features_text @ text_cache_keys + test_features @ im_cache_keys + test_features @ text_cache_keys)/3
        affinity += affinity2

        few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        zero_logits = 100. * test_features @ clip_weights
        TIDEA_logits = zero_logits + few_logits * alpha
        acc = cls_acc(TIDEA_logits, test_labels)

        print("**** IDEA-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
            torch.save(adapter2, cfg['cache_dir'] + "/best_F_2_" + str(cfg['shots']) + "shots.pt")
    
    adapter = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    adapter2 = torch.load(cfg['cache_dir'] + "/best_F_2_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, IDEA-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_theta, best_beta, best_alpha = search_hp_2a(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter, adapter2=adapter2)

    print("\n-------- Evaluating on the test set. --------")

    test_features_text = adapter(test_features)
    affinity2 = adapter2(test_features)
    affinity = best_theta* (test_features_text + test_features) @ text_cache_keys + (1-best_theta) * test_features @ im_cache_keys
    affinity += affinity2
    few_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    TIDEA_logits = zero_logits + few_logits * best_alpha
    acc = cls_acc(TIDEA_logits, test_labels)
    print("**** IDEA-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False, dataset_name=cfg['dataset'])
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False, dataset_name=cfg['dataset'])

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False, dataset_name=cfg['dataset'])
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True, dataset_name=cfg['dataset'])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    im_cache_keys, text_cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ IDEA-Adapter ------------------------------------------
    run_IDEA(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # ------------------------------------------ IDEA-Adapter-F ------------------------------------------
    run_TIDEA(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)

if __name__ == '__main__':
    main()