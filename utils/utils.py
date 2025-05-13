from tqdm import tqdm
import torch
import clip
from prompt import descriptions

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc



def clip_classifier(classnames, clip_model):
    with torch.no_grad():
        texts = [descriptions[classname] for classname in classnames]
        print("clip_classifier_info:")
        print("=====================")
        print(texts)
        texts = clip.tokenize(texts).cuda()  
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) 
        clip_weights = torch.stack([embedding for embedding in class_embeddings], dim=1).cuda()
    return clip_weights



def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels