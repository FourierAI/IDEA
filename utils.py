from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        im_cache_keys = []
        text_cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_im_features = []
                train_text_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target, dess) in enumerate(tqdm(train_loader_cache)):
                    texts = clip.tokenize(dess, truncate=True).cuda()
                    # prompt ensemble for ImageNet
                    text_features = clip_model.encode_text(texts)
                    train_text_features.append(text_features)

                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_im_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                im_cache_keys.append(torch.cat(train_im_features, dim=0).unsqueeze(0))
                text_cache_keys.append(torch.cat(train_text_features, dim=0).unsqueeze(0))

        im_cache_keys = torch.cat(im_cache_keys, dim=0).mean(dim=0)
        im_cache_keys /= im_cache_keys.norm(dim=-1, keepdim=True)
        im_cache_keys = im_cache_keys.permute(1, 0)

        text_cache_keys = torch.cat(text_cache_keys, dim=0).mean(dim=0)
        text_cache_keys /= text_cache_keys.norm(dim=-1, keepdim=True)
        text_cache_keys = text_cache_keys.permute(1, 0)

        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return im_cache_keys, text_cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target, des) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)
        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels

def search_hp_2(cfg, im_cache_keys, text_cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        theta_list = [0.05*i for i in range(1,20)]

        best_acc = 0
        best_theta, best_beta, best_alpha = 0, 0, 0
        for theta in theta_list:
            for beta in beta_list:
                for alpha in alpha_list:
                    if adapter:
                        features_text = adapter(features)
                        affinity = theta*(features_text+features) @ text_cache_keys + (1-theta)*features @ im_cache_keys
                    else:
                        cache_keys = (1-theta)*im_cache_keys+theta*text_cache_keys
                        affinity = features @ cache_keys

                    few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                    zero_logits = 100. * features @ clip_weights
                    TIDEA_logits = zero_logits + few_logits * alpha
                    acc = cls_acc(TIDEA_logits, labels)
                
                    if acc > best_acc:
                        print("New best setting, theta: {:.2f}, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(theta, beta, alpha, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                        best_theta = theta

        print("\nAfter searching, the best accuarcy: {:.2f}.\n, best theta: {:.2f}, best beta: {:.2f}, best alpha: {:.2f}".format(best_acc, best_theta, best_beta, best_alpha))

    return best_theta, best_beta, best_alpha


def search_hp_2a(cfg, im_cache_keys, text_cache_keys, cache_values, features, labels, clip_weights, adapter=None, adapter2 = None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        theta_list = [0.1*i for i in range(1,10)]

        best_acc = 0
        best_theta, best_beta, best_alpha = 0, 0, 0
        for theta in theta_list:
            for beta in beta_list:
                for alpha in alpha_list:
                    if adapter:
                        features_text = adapter(features)
                        affinity = theta*(features_text+features) @ text_cache_keys + (1-theta)*features @ im_cache_keys
                        affinity2 = adapter2(features)
                        affinity += affinity2
                    else:
                        cache_keys = (1-theta)*im_cache_keys+theta*text_cache_keys
                        affinity = features @ cache_keys

                    few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                    zero_logits = 100. * features @ clip_weights
                    TIDEA_logits = zero_logits + few_logits * alpha
                    acc = cls_acc(TIDEA_logits, labels)
                
                    if acc > best_acc:
                        print("New best setting, theta: {:.2f}, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(theta, beta, alpha, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                        best_theta = theta

        print("\nAfter searching, the best accuarcy: {:.2f}.\n, best theta: {:.2f}, best beta: {:.2f}, best alpha: {:.2f}".format(best_acc, best_theta, best_beta, best_alpha))

    return best_theta, best_beta, best_alpha


class Cross_Attention(nn.Module):
    def __init__(self, x1_dim, x2_dim, emb_dim):
        super(Cross_Attention, self).__init__()

        self.Wq = nn.Linear(x1_dim, emb_dim)
        self.Wk = nn.Linear(x2_dim, emb_dim)
        self.Wv = nn.Linear(x2_dim, emb_dim)

    def forward(self, x1, x2):
        Q = self.Wq(x1)
        K = self.Wk(x2)
        V = self.Wv(x2)

        # 获取特征维度
        d_k = Q.size(-1)  # d_k 是 Query 和 Key 的特征维度

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和值
        out = torch.matmul(attention_weights, V)
        
        return out
    
if __name__ == "__main__":
    att = Cross_Attention(100, 200, 50)
    x1 = torch.randn(2,3,100)
    x2 = torch.randn(2,1,200)
    out = att(x1,x2)
    print(out)