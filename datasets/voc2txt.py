import os.path as osp
import sys
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection():
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                image_sets=[('2007', 'val')],
                target_transform=VOCAnnotationTransform()):
        self.root = root
        self.image_set = image_sets
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def voc2txt(self):
        imgdict = {}
        labeldict = {}
        for i, (rootpath, image_name) in enumerate(self.ids):
            img_path = self._imgpath % (rootpath, image_name)
            img = cv2.imread(img_path)
            height, width, channels = img.shape
            target = ET.parse(self._annopath % (rootpath, image_name)).getroot()
            if self.target_transform is not None:
                target = self.target_transform(target, width, height)
            labeldict[image_name] = target
            imgdict[image_name] = img_path
        return imgdict, labeldict



if __name__ == '__main__':
    import numpy as np
    VOC = VOCDetection(root='/home/robin/datasets/voc/VOCdevkit')
    imgdict, labeldict = VOC.voc2txt()
    labelspath = osp.join('data/custom/labels', '%s.txt')
    with open ('data/custom/trainval.txt', 'w+') as f:
        for key in imgdict.keys():
            img_path = imgdict[key]
            img_path = img_path.split('/')[-1]
            imagepath = osp.join('data/custom/images',img_path)
            f.write(imagepath + '\n')
    
    for key in labeldict.keys():
        boxes = labeldict[key]
        labels = np.zeros((len(boxes), 5))
        labelpath = labelspath % key
        boxes = np.array(boxes)
        labels[:,0] = boxes[:,0]
        labels[:,1] = (boxes[:,1] + boxes[:,3])/2
        labels[:,2] = (boxes[:,2] + boxes[:,4])/2
        labels[:,3] = abs(boxes[:,3] - boxes[:,1])
        labels[:,4] = abs(boxes[:,4] - boxes[:,2])
        np.savetxt(labelpath, labels, fmt=' '.join(['%i'] + ['%1.4f']*4))
    
    with open('data/custom/classses.namees', 'w+') as f:
        for name in VOC_CLASSES:
            f.write(name + '\n')
