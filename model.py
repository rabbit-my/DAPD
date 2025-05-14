import torch
import torch.nn.functional as F
from utils.utils import *
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, \
    load_lora
from torch import nn
from prompt import descriptions




def evaluate_lora(
        args,
        logit_scale,
        clip_model,
        loader,
        dataset,
        prototypes=None,
        alpha=0.5,
        use_prototypes_only=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.eval()

    with torch.no_grad():
        texts = [descriptions[classname] for classname in dataset.classnames]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            text_tokens = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(text_tokens)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    tot_samples = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_logits = (image_features @ text_features.t()) * logit_scale
            if prototypes is not None:
                print("prototypes is not None")
                proto_logits = (image_features @ prototypes.to(image_features.dtype).t()) * logit_scale

                final_logits = proto_logits if use_prototypes_only else alpha * text_logits + (1 - alpha) * proto_logits
            else:
                final_logits = text_logits

            bs = target.size(0)
            acc += cls_acc(final_logits, target) * bs
            tot_samples += bs

    acc /= tot_samples
    return acc



def evaluate_lora_plot(args, logit_scale, clip_model, loader, dataset, prototypes=None, alpha=0.5, use_prototypes_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.eval()

    y_true, y_pred = [], [] 

    with torch.no_grad():
        texts = [descriptions[classname] for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            text_tokens = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(text_tokens)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    tot_samples = 0

    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_logits = (image_features @ text_features.t()) * logit_scale
            if prototypes is not None:
                proto_logits = (image_features @ prototypes.to(image_features.dtype).t()) * logit_scale
                final_logits = proto_logits if use_prototypes_only else alpha * text_logits + (1 - alpha) * proto_logits
            else:
                final_logits = text_logits

            bs = target.size(0)
            acc += cls_acc(final_logits, target) * bs
            tot_samples += bs

            y_true.extend(target.cpu().numpy())
            y_pred.extend(final_logits.argmax(dim=1).cpu().numpy())

    acc /= tot_samples
    return acc



def run_model(args, clip_model, logit_scale, dataset,
             train_loader, val_loader, test_loader, xiangya_loader,huaxi_test_loader):
    
    VALIDATION = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    textual_features = clip_classifier(dataset.classnames, clip_model)

    feature_dim = textual_features.shape[0]

    print(feature_dim)

    val_features, val_labels = pre_load_features(clip_model, val_loader)
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    clip_logits = logit_scale * (test_features @ textual_features)
    zs_acc = cls_acc(clip_logits, test_labels)
    print(f"Zero-shot CLIP's test accuracy: {zs_acc:.2f}\n")

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.to(device)


    mark_only_lora_as_trainable(clip_model)

    num_classes = len(dataset.classnames)
    prototypes = torch.nn.Parameter(F.normalize(
        torch.randn(num_classes, feature_dim, device=device, dtype=torch.float32), p=2, dim=1
    ))

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(
        list(get_lora_parameters(clip_model)) + [prototypes],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0

    label_smoothing = 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print("[Training LoRA + Prototypes as classifier] ...")
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0.0
        loss_epoch = 0.0
        tot_samples = 0

        for (images, target) in train_loader:
            texts = [descriptions[classname] for classname in dataset.classnames]
            images, target = images.to(device), target.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

                if args.encoder in ['text', 'both']:
                    text_tokens = clip.tokenize(texts).to(device)
                    class_embeddings = clip_model.encode_text(text_tokens)
                    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                else:
                    text_features = textual_features.to(device)

                if args.encoder in ['vision', 'both']:
                    image_features = clip_model.encode_image(images)
                else:
                    with torch.no_grad():
                        image_features = clip_model.encode_image(images)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                clip_logits = logit_scale * (image_features @ text_features.t())
                # clip_ce_loss = F.cross_entropy(clip_logits, target)
                clip_ce_loss = criterion(clip_logits, target)

                prototype_logits = logit_scale * (image_features @ prototypes.t())
                # proto_ce_loss = F.cross_entropy(prototype_logits, target)
                proto_ce_loss = criterion(prototype_logits, target)

                normalized_text_features = F.normalize(text_features, dim=-1)  # [5, 512]
                normalized_prototypes = F.normalize(prototypes, dim=-1)  # [5, 512]


                from geomloss import SamplesLoss

                ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.02)

                ot_loss = ot_loss_fn(normalized_prototypes, normalized_text_features)


                total_loss = 0.5 * clip_ce_loss + 0.5 * proto_ce_loss + 0.4 * ot_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = target.size(0)

            # Calculate accuracy for prototype_logits
            prototype_acc = cls_acc(prototype_logits, target)
            acc_train += prototype_acc * bs

            # Calculate accuracy for clip_logits
            clip_acc = cls_acc(clip_logits, target)

            # Update loss
            loss_epoch += total_loss.item() * bs
            tot_samples += bs

            # Print detailed information
            print(f"[Iteration {count_iters}] "
                  f"clip_ce_loss={clip_ce_loss.item():.4f}, "
                  f"proto_ce_loss={proto_ce_loss.item():.4f}, "
                  f"ot_loss={ot_loss.item():.4f}, "
                  f"total_loss={total_loss.item():.4f}, "
                  f"prototype_acc={prototype_acc:.4f}, "
                  f"clip_acc={clip_acc:.4f}")

            count_iters += 1
            if count_iters >= total_iters:
                break

        if tot_samples > 0:
            acc_train /= tot_samples
            loss_epoch /= tot_samples

        current_lr = scheduler.get_last_lr()[0]
        print(f"Iter:{count_iters}/{total_iters}, LR:{current_lr:.6f}, "
              f"Train Acc:{acc_train:.4f}, Train Loss:{loss_epoch:.4f}")

        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, logit_scale, clip_model, val_loader, dataset, prototypes=prototypes)
            print(f"**** Val accuracy: {acc_val:.2f}. ****\n")

    print("[Final Evaluation on Test Set]")

    acc_test= evaluate_lora_plot(args, logit_scale, clip_model, test_loader, dataset, prototypes=prototypes)
    print(f"Final Test accuracy: {acc_test:.2f}\n")

    acc_test_e1= evaluate_lora_plot(args, logit_scale, clip_model, xiangya_loader, dataset, prototypes=prototypes)
    print(f"Xiangya Test accuracy: {acc_test_e1:.2f}\n")


    acc_test_huaxi= evaluate_lora_plot(args, logit_scale, clip_model, huaxi_test_loader, dataset, prototypes=prototypes)
    print(f"Huaxi Test accuracy: {acc_test_huaxi:.2f}\n")




    if args.save_path is not None:
        save_lora(args, list_lora_layers)
        print(f"LoRA weights have been saved to {args.save_path}.")
    return
