import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from maskrcnn_benchmark.data.datasets import ImageNetClsDataset
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms

import numpy as np, os
from scipy import ndimage

import multiprocessing as mp
from functools import lru_cache
from copy import deepcopy

rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
if rank > 0: exit()

split = 'train'
# split = 'test'


model_cpu = models.resnet101(pretrained=True).eval()
# model_cpu = models.resnet50(pretrained=True).eval()

# Set the number of workers and gpus here for parallel compute:
num_workers = 8
num_gpu = 8
local_model = [None, None, None]

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

@lru_cache(10)
def get_transform(short_side=288):
    return T.Compose([T.Resize(short_side, interpolation=2),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      make_divisible,
                      # lambda x: make_divisible(x, s=64),
                     ])


# @lru_cache(100000)
def compute_cam_twoscale(i, scale=288, flip=True, **kwargs):
    if not local_model[0]:
        dev = i % num_gpu
        model_gpu = model_cpu.to(dev)
        local_model[0] = model_gpu
        local_model[1] = dev
        data = ImageNetClsDataset('datasets', split, proposals=0)
        local_model[2] = data
        print(i, dev, os.getpid(), id(model_gpu), data)
    model, dev, data = local_model
    ####

    img, box, _ = data[i]
    img_size = data.get_img_info(i)

    # Get CAM at scale
    input = get_transform(scale)(img)[None].to(dev)
    if flip:
        input = torch.cat([input, torch.flip(input, dims=(-1,))], 0)
        featmap, _ = forward(model, input)
        featmap = (featmap[:1] + torch.flip(featmap[1:], dims=(-1,))) * 0.5
    else:
        featmap, _ = forward(model, input)

    # Get CAM at scale*2
    input = get_transform(scale*2)(img)[None].to(dev)
    if flip:
        input = torch.cat([input, torch.flip(input, dims=(-1,))], 0)
        featmap2, _ = forward(model, input)
        featmap2 = (featmap2[:1] + torch.flip(featmap2[1:], dims=(-1,))) * 0.5
    else:
        featmap2, _ = forward(model, input)

    label = box.extra_fields['label']

    # Normalize the CAMs
    cam = conv_cam(model, featmap, label)[0,0]
    _min, _max = cam.min(), cam.max()
    cam = (cam - _min) / (_max - _min)

    cam2 = conv_cam(model, featmap2, label)[0,0]
    _min, _max = cam2.min(), cam2.max()
    cam2 = (cam2 - _min) / (_max - _min)

    # Average the CAMs of two scales
    cam = (F.interpolate(cam[None,None], cam2.shape[-2:])[0,0] + cam2) * 0.5
    return (scale*2, 1.0, label, cam), img_size



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



def compute_cam_box(i, k=4,
                    cam_tuple=None,
                    ths=[0.2,0.3,0.4,0.5],
                    bias=0.22,
                    **kwargs):
    if cam_tuple is None:
        cam_tuple, img_size = compute_cam_twoscale(i, k=4, **kwargs)
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



from tqdm import tqdm

data = ImageNetClsDataset('datasets', split, proposals=0)
n = len(data)

boxes0 = [None] * n
boxes1 = [None] * n
boxes2 = [None] * n
boxes3 = [None] * n
boxes4 = [None] * n
boxes5 = [None] * n

# preload_cam = False
# path = f'datasets/imagenet2012Full/{split}.cam.npz'
# if os.path.isfile(path):
#     print (f'{path} exists... preload')
#     npz = np.load(path, allow_pickle=True)
#     cams = npz['cams']
#     sizes = npz['sizes']
#     preload_cam = True
# assert preload_cam

num_images = 0
num_boxes = np.array([0] * 6)

pbar = tqdm(total=n, mininterval=5)
def update(args):
    global num_images, num_boxes
    i, b0,b1,b2,b3,b4,b5 = args
    pbar.update(1)
    boxes0[i] = b0
    boxes1[i] = b1
    boxes2[i] = b2
    boxes3[i] = b3
    boxes4[i] = b4
    boxes5[i] = b5
    num_images += 1
    num_boxes[0] += len(b0)
    num_boxes[1] += len(b1)
    num_boxes[2] += len(b2)
    num_boxes[3] += len(b3)
    num_boxes[4] += len(b4)
    num_boxes[5] += len(b5)
    if i % 1000 == 0:
        print(f'avg boxes per image: {np.round(num_boxes / num_images, 3)}')

ths = [0.2,0.3,0.4,0.5] 
# ths = [0.18,0.28,0.38,0.48]
# ths = [0.22,0.32,0.42,0.52]
thstr = '3'


# In case of unexpected error, use traceback to print more info.
import traceback
def compute_cam(i):
    try:
        p = compute_cam_box(i, flip=True, ths=ths)
        p0 = p
        p1 = boxlist_nms(p, 0.9)
        p2 = boxlist_nms(p, 0.8)
        p3 = boxlist_nms(p, 0.7)
        p4 = boxlist_nms(p, 0.6)
        p5 = boxlist_nms(p, 0.5)
        return (i, p0.bbox.numpy(), p1.bbox.numpy(), p2.bbox.numpy(), p3.bbox.numpy(), p4.bbox.numpy(), p5.bbox.numpy())
    except:
        traceback.print_exc()


with mp.Pool(num_workers) as p:
    for i in range(n):
        r = p.apply_async(compute_cam, args=(i,), callback=update)
        if i % (num_workers*4) == 0:
            r.wait()
    p.close()
    p.join()

pbar.close()


# if not preload_cam:
#     np.savez_compressed(f'datasets/imagenet2012Full/{split}.cam.npz', cams=cams, sizes=np.stack(sizes))

def save(path, boxes):
    np.savez_compressed(path, boxes=np.array(boxes, dtype=np.object))
    print (path, len(boxes))

save(f'datasets/imagenet2012Full/{split}.cam{thstr}xyxy.npz', boxes0)
save(f'datasets/imagenet2012Full/{split}.cam{thstr}nms9xyxy.npz', boxes1)
save(f'datasets/imagenet2012Full/{split}.cam{thstr}nms8xyxy.npz', boxes2)
save(f'datasets/imagenet2012Full/{split}.cam{thstr}nms7xyxy.npz', boxes3)
save(f'datasets/imagenet2012Full/{split}.cam{thstr}nms6xyxy.npz', boxes4)
save(f'datasets/imagenet2012Full/{split}.cam{thstr}nms5xyxy.npz', boxes5)
