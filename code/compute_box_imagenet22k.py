import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
# from maskrcnn_benchmark.data.transforms import Resize, ToTensor, Normalize
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, cat_boxlist, boxlist_iou
from maskrcnn_benchmark.data.datasets.tsv_dataset import TSVDataset
from scipy import ndimage
# from functools import lru_cache
import os, sys, time, multiprocessing as mp, numpy as np, pickle
from tqdm import trange, tqdm


rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))

model_path = 'pretrained_models/model_r50_imagenet22k.old_format.pt'
print(f'rank={rank}/{world_size}, loading pretrained classifier from {model_path}')

model_cpu = models.resnet50(pretrained=False, num_classes=21841).eval()
checkpoint = torch.load(model_path)
print(
    model_cpu.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()}, strict=True)
    )

num_gpu = 4
dev = rank % num_gpu
local_model = [model_cpu.to(dev), dev]


@torch.no_grad()
def forward(self, x, clogits=None):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    if clogits:
        clogits = F.linear(x.mean([2,3]), self.fc.weight, self.fc.bias)
    return x, clogits

@torch.no_grad()
def conv_cam(model, x, label=None):
    if label is not None:
        cam = F.conv2d(x, model.fc.weight[label:label+1,:,None,None], model.fc.bias[label:label+1])
    else:
        cam = F.conv2d(x, model.fc.weight[:,:,None,None], fc.bias)
    return cam


def make_divisible(x, s=32):
    h, w = x.shape[1:]
    hs, ws = (h+s-1)//s*s, (w+s-1)//s*s
    # since image is already normalized, pad 0
    # x = F.pad(x, (s,ws-w,s,hs-h), 'constant', value=0)
    x = F.pad(x, (0,ws-w,0,hs-h), 'constant', value=0)
    return x


# def get_transform(short_side=288):
def get_transform():
    short_side = 288
    t1 = T.Compose([T.Resize(short_side, interpolation=2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    make_divisible,
                   ])
    short_side = 288 * 2
    t2 = T.Compose([T.Resize(short_side, interpolation=2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    make_divisible,
                   ])
    def transform(img, target):
        img1 = t1(img)
        img2 = t2(img)
        return (img1, img2), target.extra_fields["labels"][0].item() - 1
    return transform


# @lru_cache(100000)
def compute_cam_twoscale(imgs, label, scale=288, flip=True, **kwargs):
    model, dev = local_model
    ####

    img1, img2 = imgs

    input = img1.to(dev)
    if flip:
        input = torch.cat([input, torch.flip(input, dims=(-1,))], 0)
        featmap, _ = forward(model, input)
        featmap = (featmap[:1] + torch.flip(featmap[1:], dims=(-1,))) * 0.5
    else:
        featmap, _ = forward(model, input)

    input = img2.to(dev)
    if flip:
        input = torch.cat([input, torch.flip(input, dims=(-1,))], 0)
        featmap2, _ = forward(model, input)
        featmap2 = (featmap2[:1] + torch.flip(featmap2[1:], dims=(-1,))) * 0.5
    else:
        featmap2, _ = forward(model, input)

    label = label.item()

    cam = conv_cam(model, featmap, label)[0,0]
    _min, _max = cam.min(), cam.max()
    cam = (cam - _min) / (_max - _min)

    cam2 = conv_cam(model, featmap2, label)[0,0]
    _min, _max = cam2.min(), cam2.max()
    cam2 = (cam2 - _min) / (_max - _min)

    cam = (F.interpolate(cam[None,None], cam2.shape[-2:])[0,0] + cam2) * 0.5
    return (scale*2, 1.0, label, cam)



def moments(m, gx, gy):
    M = m.sum()
    # center of mass
    cx, cy = (m * gx).sum() / M, (m * gy).sum() / M
    # moment of inertia
    Ix, Iy = (m * (gx - cx)**2).sum() / M, (m * (gy - cy)**2).sum() / M
    return cx.item(), cy.item(), Ix.item(), Iy.item()

def xyxy_by_moments(cam):
    gy, gx = torch.meshgrid(torch.arange(cam.shape[0], dtype=torch.float32), torch.arange(cam.shape[1], dtype=torch.float32))
    cx, cy, Ix, Iy = moments(cam, gx, gy)
    tau = 1.0 #0.99
    w, h = max(1.0,np.sqrt(Ix * 12)*tau), max(1.0, np.sqrt(Iy * 12)*tau)
    # w, h = np.sqrt(Ix * 12)*tau, np.sqrt(Iy * 12)
    # x1, y1 = cx+0.25 - w/2, cy+0.5 - h/2
    x1, y1 = cx+0.5 - w/2, cy+0.5 - h/2
    x2, y2 = x1+w, y1+h
    return x1,y1,x2,y2



def compute_cam_box(imgs, label, img_size,
                    k=4,
                    ths=[0.2,0.3,0.4,0.5],
                    bias=0.22,
                    **kwargs):
    cam_tuple = compute_cam_twoscale(imgs, label, k=4, **kwargs)
    cam0 = cam_tuple[-1].cpu()

    # img_size = data.get_img_info(i)
    img_size = (img_size['width'], img_size['height'])

    # ry, rx = float(size[0]) / cam.shape[0], float(size[1]) / cam.shape[1]
    # assert rx == 32 and ry == 32
    rx = ry = 32
    r = float(min(img_size)) / cam_tuple[0]
    ry, rx = ry * r, rx * r

    boxes = []
    for th in ths:
        label_image = cam0 > th
        # cam = (cam0 + 0.22) * (cam > 0.22)
        cam = (cam0 + bias) * label_image

        label_image, max_label = ndimage.label(label_image)
        object_slices = ndimage.find_objects(label_image, max_label)
        if not len(object_slices): continue

        # at most k boxes
        areas = [(obj[0].stop-obj[0].start)*(obj[1].stop-obj[1].start) for obj in object_slices]
        ind = np.argsort(areas)[::-1]
        a0 = areas[ind[0]] * 0.5
        kk = 0
        for o in ind:
            obj, area = object_slices[o], areas[o]
            if area < a0 or kk == k: break
            kk += 1

            mask = np.zeros_like(cam)
            mask[obj] = 1.0

            cam_o = cam * mask

            x1,y1,x2,y2 = xyxy_by_moments(cam_o)

            if (x2-x1)*(y2-y1) > 0.01 or not boxes:
                y1,x1,y2,x2 = y1*ry,x1*rx,y2*ry,x2*rx
                b = np.r_[np.clip(x1, 0, img_size[0]-1),
                          np.clip(y1, 0, img_size[1]-1),
                          np.clip(x2, 0, img_size[0]-1),
                          np.clip(y2, 0, img_size[1]-1)]
                boxes.append(b)
    boxes = np.stack(boxes)

    p = BoxList(boxes, img_size)
    p.extra_fields['scores'] = (p.bbox[:,2:] - p.bbox[:,:2]).prod(1)
    p.extra_fields['scores'] = p.extra_fields['scores'] / p.extra_fields['scores'][0]
    p.extra_fields['labels'] = torch.full((len(p),), cam_tuple[2], dtype=torch.int)
    return p


def save(path, boxes):
    # np.savez_compressed(path, boxes=np.array(boxes, dtype=np.object))
    # with open(path, 'wb') as f:
    #     pickle.dump(boxes, f)
    torch.save(boxes, path)
    print('saved to ', path, len(boxes))


# split = 'trainval'
os.makedirs('datasets/TaxImageNet22kSplit64', exist_ok=True)
data_temp = f'./TaxImageNet22kSplit64/trainval_%d_64'
path_temp = f'datasets/TaxImageNet22kSplit64/trainval_%d_64.%s.pth'

# Since the ImageNet22k (14M) is very large, we have to run inference in a
# distributed manner. Specifically, the data is split into 64 chunks. If there
# are 4 GPUs, then GPU 0 gets split indices 0, 4, 8, ...

# If a specific dataset_idx is given in argv, process that split only.
if len(sys.argv) > 1:
    dataset_idx = int(sys.argv[1])
    print(f'processing dataset_idx {dataset_idx}...')

    data = TSVDataset(data_temp % dataset_idx, transforms=get_transform())

    N = len(data)
    n = N // world_size
    n0 = rank * n
    if rank == world_size-1: n += N % world_size
    print(f'total {N} images, rank {rank} gets {n0}:{n0+n}')

    boxes0 = [None] * n
    boxes2 = [None] * n

    class RangeSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, n0, n):
            self.dataset = dataset
            self.n0 = n0
            self.n = n
        def __getitem__(self, idx):
            return self.dataset[self.n0 + idx]
        def __len__(self):
            return n
        def get_img_info(self, idx):
            return self.dataset.get_img_info(self.n0 + idx)

    data = RangeSubset(data, n0, n0)
    data_loader = DataLoader(data, num_workers=2, pin_memory=True)

    pbar = tqdm(total=n, mininterval=5)
    i = 0
    for imgs, label, _ in data_loader:
        pbar.update()
        p = compute_cam_box(imgs, label, data.get_img_info(i))
        p0 = p
        p2 = boxlist_nms(p, 0.8)
        boxes0[i] = p0
        boxes2[i] = p2
        i += 1
    pbar.close()

    save(path_temp % (dataset_idx, 'cam1xyxy') + f'.rank{rank}', boxes0)
    save(path_temp % (dataset_idx, 'cam1nms8xyxy') + f'.rank{rank}', boxes2)

    if rank == world_size-1:
        print(f'\nrank={rank}, merge results:')
        time.sleep(10.0)
        for v, my_boxes in zip(['cam1xyxy', 'cam1nms8xyxy'], [boxes0, boxes2]):
            all_boxes = []
            for i in range(world_size-1):
                path = path_temp % (dataset_idx, v) + f'.rank{i}'
                print(f'loading {path}...')
                if not os.path.isfile(path):
                    while not os.path.isfile(path):
                        time.sleep(1.0)
                    time.sleep(15.0)
                b = torch.load(path)
                all_boxes.extend(b)
            all_boxes.extend(my_boxes)
            save(path_temp % (dataset_idx, v), all_boxes)

    exit(0)

# Process all the data splits.
import traceback
try:
    for dataset_idx in range(rank, 64, world_size):
        if os.path.isfile(path_temp % (dataset_idx, 'cam1xyxy')):
            continue
        if os.path.isfile((path_temp % (dataset_idx, 'cam1xyxy'))[:-3] + 'pkl'):
            path = path_temp % (dataset_idx, 'cam1xyxy')
            pkl = path[:-3] + 'pkl'
            with open(pkl, 'rb') as f:
                z = pickle.load(f)
            save(path, z)
            path = path_temp % (dataset_idx, 'cam1nms8xyxy')
            pkl = path[:-3] + 'pkl'
            with open(pkl, 'rb') as f:
                z = pickle.load(f)
            save(path, z)
            continue

        data = TSVDataset(data_temp % dataset_idx, transforms=get_transform())
        data_loader = DataLoader(data, num_workers=2, pin_memory=True)

        n = len(data)
        print(f'rank {rank} gets {n} images from {data_temp % dataset_idx}')

        boxes0 = [None] * n
        # boxes1 = [None] * n
        boxes2 = [None] * n
        # boxes3 = [None] * n
        # boxes4 = [None] * n

        pbar = tqdm(total=n, mininterval=5)
        for imgs, label, i in data_loader:
            pbar.update()
            p = compute_cam_box(imgs, label, data.get_img_info(i))
            p0 = p
            # p1 = boxlist_nms(p, 0.9)
            p2 = boxlist_nms(p, 0.8)
            # p3 = boxlist_nms(p, 0.7)
            # p4 = boxlist_nms(p, 0.6)
            # return (i, p0.bbox.numpy(), p1.bbox.numpy(), p2.bbox.numpy(), p3.bbox.numpy(), p4.bbox.numpy())
            boxes0[i] = p0
            boxes2[i] = p2
        pbar.close()

        save(path_temp % (dataset_idx, 'cam1xyxy'), boxes0)
        save(path_temp % (dataset_idx, 'cam1nms8xyxy'), boxes2)
except:
    traceback.print_exc()

# Merge the results into one pkl file.
if rank == world_size-1:
    print(f'\nrank={rank}, merge results:')
    time.sleep(10.0)
    for v, my_boxes in zip(['cam1xyxy', 'cam1nms8xyxy'], [boxes0, boxes2]):
        all_boxes = []
        for i in range(world_size-1):
            path = f'datasets/TaxImageNet22kSplit64/{split}.{v}.rank{i}.pkl'
            print(f'loading {path}...')
            if not os.path.isfile(path):
                while not os.path.isfile(path):
                    time.sleep(1.0)
                time.sleep(20.0)
            # b = np.load(path, allow_pickle=True)['boxes']
            with open(path, 'rb') as f:
                b = pickle.load(f)
            all_boxes.append(b)
        all_boxes.append(my_boxes)
        # np.concatenate(all_boxes)
        save(f'datasets/TaxImageNet22kSplit64/{split}.{v}.pkl', sum(all_boxes, []))
