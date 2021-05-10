import torch
import io, os, numpy as np
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from base64 import decodebytes
from copy import deepcopy
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.concat_dataset import ConcatDataset

# import torch.multiprocessing as mp
# from functools import partial


class TSVDataset(torch.utils.data.Dataset):
    def __init__(self, filename, boxes_fn=None, downsample=None, transforms=None,
                 remove_images_without_annotations=False, shuffle=False, delay_loading_boxes=False):
        self.transforms = transforms

        self.fname = filename + '.tsv'
        if not os.path.isfile(self.fname):
            filename = filename[filename.find('/')+1:]  # go back to root
            self.fname = filename + '.tsv'

        self.pid = -1
        # self.ensure_file_open()

        boxes = None
        if boxes_fn:
            self.boxes_fn = boxes_fn
            self.delay_loading_boxes = delay_loading_boxes
            if not self.delay_loading_boxes:
                boxes = self.load_boxes()

        label_fn = filename + '.label.tsv'
        print (f'loading data: {self.fname}, label: {label_fn}...')
        with open(label_fn, 'r') as f:
            lines = f.readlines()

        lines = [l[:-1].split('\t') for l in lines]
        n = len(lines)

        if lines[0][1][0] == '[':
            # the original qd_data label: name, [{area, class, id, iscrowd, rect}]
            if not boxes_fn:
                # lines = [[l[0], None,None, eval(l[1]), None] for l in lines]
                lines = [[l[0], None,None, l[1], None] for l in lines]
            else:
                if delay_loading_boxes:
                    lines = [[l[0], None,None, None, None] for l in lines]
                else:
                    lines = [[l[0], None,None, boxes[i], None] for i, l in enumerate(lines)]

            with open(filename + '.lineidx', 'r') as f:
                for i, l in zip(range(n), f.readlines()):
                    lines[i][-1] = int(l)
            with open(filename + '.hw.tsv', 'r') as f:
                for i, l in zip(range(n), f.readlines()):
                    l = l.split()
                    lines[i][1] = int(l[1])
                    lines[i][2] = int(l[2])
            with open(filename[:filename.rfind('/')+1] + "labelmap.txt", 'r') as f:
                labels = f.read().split('\n')
                # self.label_to_idx = dict(zip(labels, range(1, len(labels)+1)))  # indices: 1...C
                self.label_to_idx = dict(zip(labels, range(1, len(labels))))  # indices: 1...C, last line is \n
                self.idx_to_label = dict(zip(range(1, len(labels)), labels))
            if delay_loading_boxes:
                num_boxes = 'DELAY'
            else:
                num_boxes = sum([len(l[3]) for l in lines])
            print (f'label format: old, num_images={len(lines)} num_boxes={num_boxes} num_labels={len(self.label_to_idx)}')
        else:
            # processed label: name, h, w, box[], offset
            lines = [(l[0], int(l[1]),int(l[2]), eval(l[3]), int(l[4])) for l in lines]
            num_boxes = sum([len(l[3]) for l in lines])
            print (f'label format: new, num_images={len(lines)} num_boxes={num_boxes}')

        # remove images without annotation
        if remove_images_without_annotations:
            lines = [l for l in lines if len(l[3])]
            print ('remove_images_without_annotations %d -> %d lines' % (n, len(lines)))

        self.lines = lines
        # self.f.seek(0, 2)
        # size = self.fimage.tell()
        # self.imglens = imglens
        self.transforms = transforms

        if downsample is not None:
            n = len(lines)
            idx = np.random.choice(n, round(n*downsample), replace=False)
            print ("%s downsampled from %d to %d" % (filename, n, len(idx)))
            self.lines = [self.lines[i] for i in idx]
            # self.imglens = [self.imglens[i] for i in idx]

        if shuffle:
            # random.shuffle(self.lines)
            self.indices = np.random.permutation(len(self.lines))
            self.lines = [self.lines[i] for i in self.indices]
            print (f"{filename} shuffled")
        # ImageFile.LOAD_TRUNCATED_IMAGES = True  # PIL issues on trainval_36_64, around index 138640+3178

    def load_boxes(self):
        print (f'loading boxes: {self.boxes_fn}...')
        # with open(self.boxes_fn, 'rb') as f:
        #     boxes = pickle.load(f)
        boxes = torch.load(self.boxes_fn)
        # num_boxes = sum([len(l) for l in boxes])
        return boxes

    def ensure_file_open(self):
        if self.pid != os.getpid():
            if self.lines[0][3] is None:
                boxes = self.load_boxes()
                if hasattr(self, 'indices'):
                    for i in range(len(self.lines)):
                        self.lines[i][3] = boxes[self.indices[i]]
                else:
                    for i in range(len(self.lines)):
                        self.lines[i][3] = boxes[i]
                print('done delayed loading boxes')
            self.pid = os.getpid()
            self.f = open(self.fname, 'rb')

    def __del__(self):
        try:
            self.f.close()
        except:
            pass

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        # file_idx = i 
        l = self.lines[i]
        self.ensure_file_open()
        self.f.seek(l[-1])
        b = self.f.readline()
        _ = b.find(b'\t')+1
        b = b[_ + b[_:].find(b'\t') + 1 :]
        # b = b[_ + b[_:].find(b'\t') + 1 : -1]
        b = io.BytesIO(decodebytes(b))
        try:
            img = Image.open(b)
        except:
            print(f"Error reading entry {i} {l}")
            # OSError: cannot identify image file <_io.BytesIO object at 0x7f317883fe60>
            # when reading one file from trainval_36_64
            l = self.lines[i] = self.lines[(i + 1) % len(self.lines)]
            self.f.seek(l[-1])
            b = self.f.readline()
            _ = b.find(b'\t')+1
            b = b[_ + b[_:].find(b'\t') + 1 :]
            b = io.BytesIO(decodebytes(b))
            img = Image.open(b)
        img.load()
        b.close()
        img = img.convert("RGB")

        # anno = l[3]
        target = self.get_groundtruth(i)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, i

    def get_img_info(self, i):
        h, w = self.lines[i][1:3]
        return {'height': h, 'width': w}

    def get_groundtruth(self, i):
        l = self.lines[i]
        h, w = l[1:3]
        if isinstance(l[3], BoxList):
            assert l[3].size == (w,h)
            # BUG/TODO: labels in BoxList is 0-based, but really should be 1-based!
            # assert not torch.any(torch.equal(l[3].extra_fields['labels'], 0))
            return deepcopy(l[3])

        if isinstance(l[3], str):
            l[3] = eval(l[3])
        anno = l[3]

        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (w,h), mode="xyxy")

        # 0 is the background, class index starts from *1*
        if len(anno) and "c" not in anno[0]:
            for obj in anno:
                obj["c"] = self.label_to_idx[obj["class"]]
        # classes = [self.label_to_idx[obj["class"]] for obj in anno]
        classes = [obj["c"] for obj in anno]
        # target.add_field("label", set(classes))
        classes = torch.as_tensor(classes, dtype=torch.int64)
        target.add_field("labels", classes)

        # target = target.clip_to_image(remove_empty=True)
        return target

    def map_class_id_to_class_name(self, i):
        return self.idx_to_label[i]


def load_split(i, sep_label=None, **kwargs):
    if sep_label:
        ds = TSVDataset(
            f'datasets/TaxImageNet22kSplit64/trainval_{i}_64',
            f'datasets/TaxImageNet22kSplit64/trainval_{i}_64.{sep_label}.pth',
            **kwargs)
    else:
        ds = TSVDataset(f'TaxImageNet22kSplit64/trainval_{i}_64', **kwargs)
    return ds

class TaxImageNet22kSplit64(ConcatDataset):
    def __init__(self, sep_label=None, max_n=64, shuffle=False, **kwargs):
        # mp.set_sharing_strategy('file_system')
        # with mp.Pool(min(24, max_n)) as pool:
        #     datasets = pool.map(partial(load_split, sep_label=sep_label, **kwargs), range(max_n))
        kwargs['remove_images_without_annotations'] = False  # not necessary
        kwargs['delay_loading_boxes'] = True

        if shuffle:
            # order = np.random.permutation(max_n)
            order = [
                46,25,59,9,2,13,16,14,7,54,40,23,47,53,15,55,61,56,20,10,52,36,30,60,
                31,17,19,43,38,51,58,1,18,34,63,26,11,41,21,12,39,27,44,29,32,45,8,35,
                33,5,37,24,0,6,3,62,50,22,48,28,42,57,49,4]
            # rand_state = np.random.RandomState(int(os.environ['OPTS_HASH']))
            # order = rand_state.permutation(max_n)
            kwargs['shuffle'] = True
            print(f"TaxImageNet22kSplit64 random order: {order}")
        else:
            order = range(max_n)
            print(f"TaxImageNet22kSplit64 sequential order max_n={max_n}")
        datasets = [load_split(i, sep_label, **kwargs) for i in order]
        super().__init__(datasets)

    def get_groundtruth(self, i):
        dataset_idx, sample_idx = super().get_idxs(i)
        return self.datasets[dataset_idx].get_groundtruth(sample_idx)

    def map_class_id_to_class_name(self, i):
        return self.datasets[0].map_class_id_to_class_name(i)
