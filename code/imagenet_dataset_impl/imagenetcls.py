import torch.utils.data
# from base64 import decodebytes
from PIL import Image
from io import BytesIO
# import mmap
import os, numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
try:
    from dlib import find_candidate_object_locations
except:
    pass
# from multiprocessing import Manager
from copy import deepcopy

# class BinaryDataset(torch.utils.data.Dataset):
class ImageNetClsDataset(torch.utils.data.Dataset):
    '''
    More efficient dataset interface with images in binary format
    
    e.g.
      img_file:   imagenet2012/train.binary
        concatenated raw JPG/PNG/... files
      label_file: imagenet2012/train.binary.label
        lines of tab separated (name, label_id, offset, size) tuples

    proposals: int. generate selective search proposals
    cache_proposals: bool. whether to cache proposals into memory
    '''
    # def __init__(self, img_file, label_file, transform=None):
    def __init__(self, root='datasets', desc='test', transforms=None,
                 proposals=0,
                 # cache_proposals=True,
                 # download_data=False,
                 **kwargs):

        desc, *extra_desc = desc.split('_')
        extra_desc = set(extra_desc)
        download_data = 'down' in extra_desc

        img_file = os.path.join(root, f'imagenet2012Full/{desc}.binary')
        label_file = os.path.join(root, f'imagenet2012Full/{desc}.binary.label')

        if download_data:
            extra_desc.remove('down')

            import fcntl
            def acquireLock():
                ''' acquire exclusive lock file access '''
                locked_file_descriptor = open('/tmp/lockfile.LOCK', 'w+')
                fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
                return locked_file_descriptor

            def releaseLock(locked_file_descriptor):
                ''' release exclusive lock file access '''
                locked_file_descriptor.close()

            acquireLock()

            img_file_local = '/dev/shm/databin'
            if not os.path.isfile(img_file_local):
                print (f'copying file to {img_file_local} ...')
                from shutil import copyfile
                copyfile(img_file, img_file_local)

            releaseLock()

        # support multiprocess
        self.fname = img_file
        self.pid = -1
        # self.ensure_file_open()
        # On the use of mmap:
        #   Pro: it's shared and doesn't seem to have read lock?
        #   Con: Turns out to be slightly slower than file
        # self.f = open(self.fname, 'rb')
        # self.f = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

        with open(label_file, 'r') as f:
            self.labels = [[int(x) for x in _.split('\t')[1:6]]
                    for _ in f.read().splitlines()]
        print (f'load {label_file}, len={len(self.labels)}')
        # assert all([len(_)==3 for _ in self.labels])
        self.transforms = transforms

        self.proposals = proposals
        # load cached proposal files
        if len(extra_desc):
            # assert proposals > 0, 'proposals must > 0 if loading boxes'
            self.proposals = max(self.proposals, 1)

            prop = extra_desc.pop()
            if 'pseudo' in prop:
                prop_file = os.path.join(root, f'imagenet2012Full/{desc}.{prop}.pth')
                print ("type=2", prop_file)
                self.cache_proposals = torch.load(prop_file)
                self.type = 2
            else:
                prop_file = os.path.join(root, f'imagenet2012Full/{desc}.{prop}.npz')
                npz = np.load(prop_file, allow_pickle=True)
                keys = list(npz.keys())
                self.cache_proposals = npz[keys[0]]
                if 'scores' in keys:
                    self.cache_scores = npz['scores']
                print ("type=1", prop_file, keys, 'bbox[0]', self.cache_proposals[0])
                self.type = 1
        # elif 'ss' in extra_desc:
        #     prop_file = os.path.join(root, f'imagenet2012Full/{desc}.prop.npz')
        #     self.cache_proposals = np.load(prop_file, allow_pickle=True)['prop']
        else:
            self.cache_proposals = None
        # self.cache_proposals = cache_proposals
        # if cache_proposals:
        #     # self.cache_proposals = [None] * len(self.labels)
        #     manager = Manager()
        #     self.cache_proposals = manager.list([None] * len(self.labels))

    def ensure_file_open(self):
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.f = open(self.fname, 'rb')
            # self.f = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def generate_proposals(self, img, i, img_size=None, label=None):
        if img_size is None:
            img_size = img.size
        elif img is not None:
            assert img_size == img.size
        if self.cache_proposals is not None:
            if self.type == 1:
                # NOTE: assert cache_proposals match the img_size!!
                boxlist = self.cache_proposals[i].reshape(-1, 4).copy()
                boxlist = BoxList(boxlist, img_size, mode='xyxy') #.convert('xyxy')
                if hasattr(self, 'cache_scores'):
                    boxlist.extra_fields['scores'] = torch.from_numpy(self.cache_scores[i].copy())
            elif self.type == 2:
                boxlist = self.cache_proposals[i]
                if boxlist.size != img_size:
                    self.cache_proposals[i] = boxlist = boxlist.resize(img_size)
                return deepcopy(boxlist)
                # return boxlist.copy_with_fields('labels')
        else:
            # np_size = (500, 500)
            if img_size[0] < img_size[1]:
                np_size = (500*img_size[0]//img_size[1], 500)
            else:
                np_size = (500, 500*img_size[1]//img_size[0])
            rects = []
            np_img = np.array(img.resize(np_size))
            # find_candidate_object_locations(np_img, rects, kvals=(400, 700, 1), min_size=5000, max_merging_iterations=50)
            find_candidate_object_locations(np_img, rects, kvals=(1000, 2000, 3), min_size=2000, max_merging_iterations=50)
            boxes = [[r.left(), r.top(), r.width(), r.height()] for r in rects]
            boxlist = BoxList(boxes, np_size, mode='xywh').convert('xyxy').resize(img.size)
            # if self.cache_proposals:
            #     self.cache_proposals[i] = boxlist
        if label is not None:
            # 0 is the background --> convert label (0-based) to labels (1-based)!
            classes = [label + 1] * len(boxlist)
            classes = torch.tensor(classes, dtype=torch.int64)
            boxlist.add_field("labels", classes)
        return boxlist

    def __getitem__(self, i):
        label, offset, size = self.labels[i][:3]
        self.ensure_file_open()
        self.f.seek(offset)
        b = BytesIO(self.f.read(size))
        # TODO: BytesIO is said to manage its own buffer, could be slow
        # b = BytesIO(self.f[offset:offset+size])
        img = Image.open(b).convert('RGB')
        img.load()
        b.close()

        if self.proposals > 0:
            boxlist = self.generate_proposals(img, i, label=label)

            # nms_thresh = 0.7
            # boxlist.add_field('scores', torch.randn(len(boxlist)))
            # boxlist = boxlist_nms(boxlist, nms_thresh=nms_thresh, max_proposals=self.proposals)
        else:
            # boxes = torch.empty(0, 4)
            # boxlist = BoxList(boxes, img.size, mode='xyxy')
            boxlist = BoxList([[0,0,*img.size]], img.size, mode='xyxy')

        if self.transforms:
            img, boxlist = self.transforms(img, boxlist)

        # boxlist.label = label
        boxlist.add_field('label', label)
        return img, boxlist, i

    def __len__(self):
        return len(self.labels)

    # def get_groundtruth(self, i):
    #     return self.labels[i][0]

    def get_img_info(self, i):
        w, h = self.labels[i][3:5]
        return {'height': h, 'width': w}

    def get_groundtruth(self, i):
        label = self.labels[i][0]
        img_size = self.labels[i][3:5]
        if self.proposals > 0:
            boxlist = self.generate_proposals(None, i, img_size=img_size, label=label)
        else:
            boxlist = BoxList([[0,0,*img_size]], img_size, mode='xyxy')
        boxlist.add_field('label', label)
        return boxlist

    # c >= 1 as c == 0 is background class
    def map_class_id_to_class_name(self, c):
        return c - 1  # self.idx_to_label[c]


# generate labels with img hw info
if __name__ == '__main__':
    from tqdm import tqdm
    # for desc in ['test', 'train']:
    for desc in ['train']:
        self = ImageNetClsDataset(desc=desc)
        with open(f'datasets/imagenet2012Full/{desc}.binary.label', 'r') as f, \
             open(f'datasets/imagenet2012Full/{desc}.binary.label+', 'w') as fo:
            for img, target, i in tqdm(self):
                # self.labels[i].append(img.size[0], img.size[1])
                # w: img.size[0], h: img.size[1]
                wh = f'\t{img.size[0]}\t{img.size[1]}\n'
                line = f.readline()
                fo.write(line.strip() + wh)

# generate proposals
if False:
    from tqdm import tqdm
    import multiprocessing as mp

    # for desc in ['test', 'train']:
    desc = 'train'

    self = ImageNetClsDataset(desc=desc, proposals=0)
    n = len(self)
    # prop = [None] * n

    def gen_prop(i):
        label, offset, size = self.labels[i]
        self.ensure_file_open()
        self.f.seek(offset)
        b = BytesIO(self.f.read(size))
        img = Image.open(b).convert('RGB')
        img.load()
        b.close()
        boxlist = self.generate_proposals(img)
        return boxlist.bbox.numpy().astype(np.int16) # save space

    with mp.Pool(32) as p:
        prop = list(tqdm(p.imap(gen_prop, range(n)), total=n))

    np.savez_compressed(f'datasets/imagenet2012Full/{desc}.prop.npz', prop=prop)

# class TSVDataset(torch.utils.data.Dataset):
#     # Efficient dataset in TSV files
#     #
#     # e.g.
#     #   img_tsv:   imagenet2012/train.tsv
#     #     lines of tab separated (name, label_id, raw img file in base64)
#     #   label_tsv: imagenet2012/train.labelline.tsv
#     #     lines of tab separated (name, label_id, offset)
#     #
#     def __init__(self, img_tsv, label_tsv, transform=None):
#         # support multiprocess
#         self.fname = img_tsv
#         self.pid = -1
#         self.ensure_file_open()

#         # f = lambda t: (int(t[1]), int(t[2]))
#         self.labels = [(int(x) for x in _.split('\t')[1:3]) for _ in open(label_tsv, 'r')]
#         self.transform = transform

#     def ensure_file_open(self):
#         if self.pid != os.getpid():
#             self.pid = os.getpid()
#             self.f = open(self.fname, 'rb')
#             # self.f = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

#     def __getitem__(self, i):
#         label, idx = self.labels[i]
#         self.ensure_file_open()
#         self.f.seek(idx)
#         line = self.f.readline()
#         # fname, label, img = line.split(b'\t')
#         fname, _, img = line.split(b'\t')
#         # if self.use_label_file:
#         #     label = self.labels[i]
#         # else:
#         #     label = int(label)
#         # img = Image.open(BytesIO(decodestring(img))).convert('RGB')
#         b = BytesIO(decodebytes(img))
#         img = Image.open(b).convert('RGB')
#         img.load()
#         b.close()
#         if self.transform: img = self.transform(img)
#         return img, label

#     def __len__(self):
#         return len(self.labels)
