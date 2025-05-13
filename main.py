
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils.build_options import set_random_seed , build_default_options
from model import run_model



def main():

    # Load config file
    args = build_default_options()
    set_random_seed(args.seed)

    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 16

    # Prepare dataset
    print("Preparing dataset.")
    train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.75, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    mixed_dataset = build_dataset(args.train_dataset, args.root_path, args.shots)

    xiangya_dataset = build_dataset(args.test_dataset_a, args.root_path, args.shots)
    huaxi_dataset = build_dataset(args.test_dataset_b, args.root_path, args.shots)

    mixed_val_loader = build_data_loader(data_source=mixed_dataset.val, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False,  num_workers=8)
    mixed_test_loader = build_data_loader(data_source=mixed_dataset.test, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False,  num_workers=8)
    
    xiangya_test_loader = build_data_loader(data_source=xiangya_dataset.train_x, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False, num_workers=8)
    huaxi_test_loader = build_data_loader(data_source=huaxi_dataset.train_x, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False, num_workers=8)


    train_loader = build_data_loader(data_source=mixed_dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)
    print(len(train_loader))

    print("Data preparation completed!")

    run_model(args, clip_model, logit_scale, mixed_dataset, train_loader, mixed_val_loader, mixed_test_loader, xiangya_test_loader, huaxi_test_loader)

if __name__ == '__main__':
    main()
