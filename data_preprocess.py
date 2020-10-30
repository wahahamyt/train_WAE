import os
import torchvision.datasets as datasets
from torch.autograd import Variable
import cv2


class Preprocessor(object):
    def __init__(self):
        self.dset_dir = '/media/x/datasets/ILSVRC2015/Data/VID'
        self.imsz = 64

        dataset_train = datasets.ImageFolder(
            os.path.join(self.dset_dir, 'train')
        )
        self.imgs = dataset_train.imgs
        dataset_val = datasets.ImageFolder(
            os.path.join(self.dset_dir, 'val')
        )
        self.imgs.append(dataset_val.imgs)
        dataset_test = datasets.ImageFolder(
            os.path.join(self.dset_dir, 'test')
        )
        self.imgs.append(dataset_test.imgs)

    def run(self):
        seg = 10000
        seg_indx = 0

        for i in range(len(self.imgs)):
            img = cv2.imread(self.imgs[i][0])
            img = cv2.resize(img,(self.imsz, self.imsz))
            if i % seg == 0:
                if ~os.path.exists('/media/x/datasets/ILSVRC2015_64/{}'.format(seg_indx)):
                    os.mkdir('/media/x/datasets/ILSVRC2015_64/{}'.format(seg_indx))
                seg_indx = seg_indx + 1

            cv2.imwrite('/media/x/datasets/ILSVRC2015_64/{0}/{1}.JPEG'.format(seg_indx-1, i), img)
            print('{0}/{1}\n'.format(i, len(self.imgs)))

if __name__ == '__main__':
    preprocessor = Preprocessor()

    preprocessor.run()
