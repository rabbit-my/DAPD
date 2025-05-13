import random
import argparse  
import numpy as np 
import torch


def build_default_options():
    parser = argparse.ArgumentParser(description="Arguments for training model.")


    parser.add_argument('--seed', default=5, type=int, help='Random seed for reproducibility')

    # dataset
    parser.add_argument('--root_path', type=str, default='/home/codebase/Yinmi/lora-my/DATA/', help='Path to the dataset')
    parser.add_argument('--train_dataset', type=str, default='mixed_center_oct', help='Train Dataset name')
    parser.add_argument('--test_dataset_a', type=str, default='xiangya_oct', help='Test Dataset name')
    parser.add_argument('--test_dataset_b', type=str, default='huaxi_oct', help='Test Dataset name')

    parser.add_argument('--shots', default=32, type=int, help='Number of shots for few-shot learning')

    # model
    parser.add_argument('--backbone', default='ViT-B/16', type=str, help='Backbone model for CLIP')

    #proto
    parser.add_argument('--proto_alpha', default=0.8, type=float, help='Weight for text-based logits in final logits computation')
    parser.add_argument('--clip_loss_weight', default=0.5, type=float, help='Weight for CLIP cross-entropy loss')
    parser.add_argument('--proto_loss_weight', default=0.5, type=float, help='Weight for prototype cross-entropy loss')
    parser.add_argument('--ot_loss_weight', default=0.8, type=float, help='Weight for optimal transport loss')

    # train
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--n_iters', default=10, type=int, help='Number of training iterations')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')

    # LoRA 
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'],
                        help='Where to put the LoRA modules in the model')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both',
                        help='Apply LoRA to text encoder, vision encoder, or both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v','o'], 
                        help='Attention matrices to apply LoRA. Options: q (query), k (key), v (value), o (output)')
    parser.add_argument('--r', default=2, type=int, help='Rank of the low-rank matrices in LoRA')
    parser.add_argument('--alpha', default=1, type=int, help='Scaling factor for LoRA (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='Dropout rate before applying LoRA')

    # save
    parser.add_argument('--save_path', default=None, type=str, help='Path to save the LoRA modules after training (None means not saving)')
    parser.add_argument('--filename', default='lora_weights', type=str, help='Filename for saving the LoRA weights (.pt extension will be added)')

    # eval
    parser.add_argument('--eval_only', default=False, action='store_true', help='Only evaluate the LoRA modules (requires save_path)')

    return parser.parse_args()

    
def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True