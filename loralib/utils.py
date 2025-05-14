import os

import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA, LinearLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


# def apply_lora(args, clip_model):
#     list_lora_layers = []
#     if args.encoder == 'text' or args.encoder == 'both':
#         indices = INDEX_POSITIONS_TEXT[args.position]
#         text_encoder = clip_model.transformer
#         for i, block in enumerate(text_encoder.resblocks):
#             print(f"Residual Attention Block {i}: {block}")
#             if i in indices:
#                 for name, submodule in block.named_children():
#                     if isinstance(submodule, nn.MultiheadAttention):
#                         new_multi_head_lora = PlainMultiheadAttentionLoRA(
#                             submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
#                         setattr(block, name, new_multi_head_lora)
#                         list_lora_layers.append(new_multi_head_lora)

#     if args.encoder == 'vision' or args.encoder == 'both':
#         indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
#         vision_encoder = clip_model.visual.transformer
#         for i, block in enumerate(vision_encoder.resblocks):
#             print(f"Residual Attention Block {i}: {block}")
#             if i in indices:
#                 for name, submodule in block.named_children():
#                     if isinstance(submodule, nn.MultiheadAttention):
#                         new_multi_head_lora = PlainMultiheadAttentionLoRA(
#                             submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
#                         setattr(block, name, new_multi_head_lora)
#                         list_lora_layers.append(new_multi_head_lora)
#     return list_lora_layers



def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():

                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)


    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():

                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

                    if name == "mlp" and isinstance(submodule, nn.Sequential):
                        for mlp_name, mlp_layer in submodule.named_children():
                            if isinstance(mlp_layer, nn.Linear):
                                lora_linear = LinearLoRA(
                                    mlp_layer,
                                    r=args.r,
                                    lora_alpha=args.alpha,
                                    fan_in_fan_out=False,
                                    dropout_rate=args.dropout_rate,
                                )
                                setattr(submodule, mlp_name, lora_linear)
                                list_lora_layers.append(lora_linear)

    return list_lora_layers


# def save_lora(args, list_lora_layers):
#     weights = {}
#     for i, layer in enumerate(list_lora_layers):
#         layer_weights = {}
#         if 'q' in args.params:
#             layer_weights['q_proj'] = {
#                 'w_lora_A': layer.q_proj.w_lora_A.data,
#                 'w_lora_B': layer.q_proj.w_lora_B.data
#             }
#         if 'k' in args.params:
#             layer_weights['k_proj'] = {
#                 'w_lora_A': layer.k_proj.w_lora_A.data,
#                 'w_lora_B': layer.k_proj.w_lora_B.data
#             }
#         if 'v' in args.params:
#             layer_weights['v_proj'] = {
#                 'w_lora_A': layer.v_proj.w_lora_A.data,
#                 'w_lora_B': layer.v_proj.w_lora_B.data
#             }
#         if 'o' in args.params:
#             layer_weights['proj'] = {
#                 'w_lora_A': layer.proj.w_lora_A.data,
#                 'w_lora_B': layer.proj.w_lora_B.data
#             }

#         weights[f'layer_{i}'] = layer_weights

#     metadata = {
#         'r': args.r,
#         'alpha': args.alpha,
#         'encoder': args.encoder,
#         'params': args.params,
#         'position': args.position
#     }

#     save_data = {
#         'weights': weights,
#         'metadata': metadata
#     }

#     # to manage names like ViT-B/16
#     backbone = args.backbone.replace('/', '').replace('-', '').lower()
#     save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
#     os.makedirs(save_dir, exist_ok=True)

#     save_path = f'{save_dir}/{args.filename}.pt'
#     torch.save(save_data, save_path)
#     print(f'LoRA weights saved to {save_path}')




def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}


        if isinstance(layer, PlainMultiheadAttentionLoRA):
            if 'q' in args.params:
                layer_weights['q_proj'] = {
                    'w_lora_A': layer.q_proj.w_lora_A.data,
                    'w_lora_B': layer.q_proj.w_lora_B.data
                }
            if 'k' in args.params:
                layer_weights['k_proj'] = {
                    'w_lora_A': layer.k_proj.w_lora_A.data,
                    'w_lora_B': layer.k_proj.w_lora_B.data
                }
            if 'v' in args.params:
                layer_weights['v_proj'] = {
                    'w_lora_A': layer.v_proj.w_lora_A.data,
                    'w_lora_B': layer.v_proj.w_lora_B.data
                }
            if 'o' in args.params:
                layer_weights['proj'] = {
                    'w_lora_A': layer.proj.w_lora_A.data,
                    'w_lora_B': layer.proj.w_lora_B.data
                }


        elif isinstance(layer, LinearLoRA):
            layer_weights['linear'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data
            }

        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

        weights[f'layer_{i}'] = layer_weights

    # 保存元信息
    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }


    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')



# def load_lora(args, list_lora_layers):
#     # to manage names like ViT-B/16
#     backbone = args.backbone.replace('/', '').replace('-', '').lower()
#     load_path = f'/home/codebase/Yinmi/lora-my/checkpoints//vitb16/mixed_center_oct/32shots/seed42/lora_weights.pt'

#     if not os.path.exists(load_path):
#         raise FileNotFoundError(f'File {load_path} does not exist.')

#     loaded_data = torch.load(load_path)

#     metadata = loaded_data['metadata']
#     if metadata['r'] != args.r:
#         raise ValueError(
#             f"r mismatch: expected {args.r}, found {metadata['r']}")
#     if metadata['alpha'] != args.alpha:
#         raise ValueError(
#             f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
#     if metadata['encoder'] != args.encoder:
#         raise ValueError(
#             f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
#     if metadata['params'] != args.params:
#         raise ValueError(
#             f"Params mismatch: expected {args.params}, found {metadata['params']}")
#     if metadata['position'] != args.position:
#         raise ValueError(
#             f"Position mismatch: expected {args.position}, found {metadata['position']}")

#     weights = loaded_data['weights']
#     for i, layer in enumerate(list_lora_layers):
#         layer_weights = weights[f'layer_{i}']
#         if 'q' in args.params and 'q_proj' in layer_weights:
#             layer.q_proj.w_lora_A.data.copy_(
#                 layer_weights['q_proj']['w_lora_A'])
#             layer.q_proj.w_lora_B.data.copy_(
#                 layer_weights['q_proj']['w_lora_B'])
#         if 'k' in args.params and 'k_proj' in layer_weights:
#             layer.k_proj.w_lora_A.data.copy_(
#                 layer_weights['k_proj']['w_lora_A'])
#             layer.k_proj.w_lora_B.data.copy_(
#                 layer_weights['k_proj']['w_lora_B'])
#         if 'v' in args.params and 'v_proj' in layer_weights:
#             layer.v_proj.w_lora_A.data.copy_(
#                 layer_weights['v_proj']['w_lora_A'])
#             layer.v_proj.w_lora_B.data.copy_(
#                 layer_weights['v_proj']['w_lora_B'])
#         if 'o' in args.params and 'proj' in layer_weights:
#             layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
#             layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

#     print(f'LoRA weights loaded from {load_path}')

def load_lora(args, list_lora_layers):
    
    load_path = f'/home/codebase/Yinmi/lora-my/checkpoints//vitb16/mixed_center_oct/32shots/seed42/lora_weights.pt'
    
    loaded_data = torch.load(load_path)
    metadata = loaded_data['metadata']

    # 确保实验参数匹配
    for key in ['r', 'alpha', 'encoder', 'params', 'position']:
        if metadata[key] != getattr(args, key):
            raise ValueError(f"{key} mismatch: expected {getattr(args, key)}, found {metadata[key]}")

    weights = loaded_data['weights']
    print("Available layers in saved weights:", weights.keys())
    print("Expected layers from list_lora_layers:", [f"layer_{i}" for i in range(len(list_lora_layers))])

    for i, layer in enumerate(list_lora_layers):  
        layer_weights = weights.get(f'layer_{i}')
        if layer_weights is None:
            raise KeyError(f"Missing layer layer_{i} in saved weights")

        # MultiheadAttention LoRA
        if isinstance(layer, PlainMultiheadAttentionLoRA):
            if 'q' in args.params and 'q_proj' in layer_weights:
                layer.q_proj.w_lora_A.data.copy_(layer_weights['q_proj']['w_lora_A'])
                layer.q_proj.w_lora_B.data.copy_(layer_weights['q_proj']['w_lora_B'])

            if 'k' in args.params and 'k_proj' in layer_weights:
                layer.k_proj.w_lora_A.data.copy_(layer_weights['k_proj']['w_lora_A'])
                layer.k_proj.w_lora_B.data.copy_(layer_weights['k_proj']['w_lora_B'])

            if 'v' in args.params and 'v_proj' in layer_weights:
                layer.v_proj.w_lora_A.data.copy_(layer_weights['v_proj']['w_lora_A'])
                layer.v_proj.w_lora_B.data.copy_(layer_weights['v_proj']['w_lora_B'])

            if 'o' in args.params and 'proj' in layer_weights:
                layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
                layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

        # Linear LoRA
        elif isinstance(layer, LinearLoRA):
            if 'linear' in layer_weights:
                layer.w_lora_A.data.copy_(layer_weights['linear']['w_lora_A'])
                layer.w_lora_B.data.copy_(layer_weights['linear']['w_lora_B'])

        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

    print(f'LoRA weights successfully loaded from {load_path}')
