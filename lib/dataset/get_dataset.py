import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.vocdataset import VOC2007
from dataset.voc07 import Voc07Dataset
from dataset.vg500dataset import VGDataset
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp

def get_datasets(args):


    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                            transforms.ToTensor()]
    test_data_transform_list =  [transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor()]
    if args.cutout:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    
    if args.remove_norm is False:
        if args.orid_norm:
            normalize = transforms.Normalize(mean=[0, 0, 0],
                                            std=[1, 1, 1])
            print("mean=[0, 0, 0], std=[1, 1, 1]")
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
        
        train_data_transform_list.append(normalize)
        test_data_transform_list.append(normalize)

    train_data_transform = transforms.Compose(train_data_transform_list)
    test_data_transform = transforms.Compose(test_data_transform_list)

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )    
    elif args.dataname == 'voc2007' or args.dataname == 'VOC2007':
        dataset_dir = args.dataset_dir
        # print(dataset_dir)
        # print('=='*30)
        # train_dataset = VOC2007(dataset_dir, phase='trainval', transform=train_data_transform)
        # val_dataset = VOC2007(dataset_dir, phase='test', transform=test_data_transform)
        dup=None
        train_dataset = Voc07Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
                                    transform = train_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
                                    dup=None)

        val_dataset = Voc07Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'), 
                                    transform = test_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
                                    dup=None)
    
    elif args.dataname == 'vg' or args.dataname == 'vg500':
        dataset_dir = args.dataset_dir
        vg_root = osp.join(dataset_dir, 'VG')
        train_dir = osp.join(vg_root,'VG_100K')
        train_list = osp.join(vg_root,'train_list_500.txt')
        test_dir = osp.join(vg_root,'VG_100K')
        test_list = osp.join(vg_root,'test_list_500.txt')
        train_label = osp.join(vg_root,'vg_category_500_labels_index.json')
        test_label = osp.join(vg_root,'vg_category_500_labels_index.json')

        train_dataset = VGDataset(train_dir, train_list, train_data_transform, train_label)
        val_dataset = VGDataset(test_dir, test_list, test_data_transform, test_label)

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
