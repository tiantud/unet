############################################################
####################       import       ####################
############################################################
import os
import pdb
import time
import json
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

from datetime import datetime
from torchvision import transforms, datasets
from collections import OrderedDict

from Segmentationdatasets import SegmentationImageFolder


############################################################
####################  global variables  ####################
############################################################

# start date
DESCRIPTION = 'normal_unet'
DATE_INFO = datetime.today().strftime('%Y%m%d_%H%M%S')

# set seed
RANDOM_SEED = 2020

# training param
LR = 0.00001
IMG_SIZE = 256
BATCH_SIZE = 25
NUM_EPOCHS = 100
NUM_WORKERS = 0             # speed up loading data, set as 0 under Windows

DATE_INFO += '_{}_e{}_lr{}'.format(DESCRIPTION, NUM_EPOCHS, LR)

# dataset path
DATASET_BLOOD = "/home/tianz/0_Workspace/grouped_dataset"
DATASET_NO_BLOOD = "/home/tianz/0_Workspace/grouped_dataset_noblood"

# model save path
SAVE_ROOT = "/home/tianz/0_Workspace/save"

# num_blood / (num_blood + num_noblood)
BLOOD_IMG_PROPORTION = 0.7

# crossvalidation steps
CROSSVAL_LIST = [
    {
        "title": "val_n1",
        "val_blood": ["SER02"],
        "val_noblood": ["SER18_NO"],
        "train_noblood": ["SER15_NO", "SER16_NO"]
    },
    {
        "title": "val_n2",
        "val_blood": ["SER01"],
        "val_noblood": ["SER16_NO"],
        "train_noblood": ["SER15_NO", "SER18_NO"]
    }, 
    {
        "title": "val_n3",
        "val_blood": ["SER03", "SER04", "SER05", "SER06", "SER07", "SER08"],
        "val_noblood": ["SER15_NO"],
        "train_noblood": ["SER16_NO", "SER18_NO"]
    }, 
    {
        "title": "val_n4",
        "val_blood": ["SER09", "SER10", "SER11", "SER12", "SER13"],
        "val_noblood": ["SER15_NO"],
        "train_noblood": ["SER16_NO", "SER18_NO"]
    },
    {
        "title": "val_n5",
        "val_blood": ["SER14", "SER15", "SER16", "SER17"],
        "val_noblood": ["SER15_NO"],
        "train_noblood": ["SER18_NO"]
    }
]

DATA_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

DATA_TRANSFORMS_NOFLIP = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

############################################################
####################    preprocessing   ####################
############################################################
def report_crossval_element(crossval_element):
    """Checks if a file is an allowed extension.
    Args:
        crossval_element (dict)                             One step of cross validation
        crossval_element.val_blood (list of strings)        Clip(in DATASET_BLOOD) belongs to validation set if folder contains any keyword in this list.
        crossval_element.val_noblood (list of strings)      Clip(in DATASET_NO_BLOOD) belongs to validation set if folder contains any keyword in this list.
        crossval_element.train_noblood (list of strings)    Hard coded no blood clips. Training set should not contains any clip which simillar to validation set.
    """
    num_blood_val, num_blood_train = 0, 0
    for fpath, _, fnames in os.walk(DATASET_BLOOD, followlinks=True):
        _, clip_id = os.path.split(fpath)

        assert len(fnames) % 2 == 0, "Missing mask image under path: {}".format(fpath)

        if any( key in clip_id for key in crossval_element["val_blood"]):
            num_blood_val += len(fnames)
        else:
            num_blood_train += len(fnames)

    num_blood_val = int(num_blood_val / 2)
    num_blood_train = int(num_blood_train / 2)

    factor = (1 - BLOOD_IMG_PROPORTION) / BLOOD_IMG_PROPORTION

    print("----------------------------------------------")
    print("Reportinging group with validation: " + ", ".join(crossval_element["val_blood"]))

    print("Number of   blood imgs in validation set: {}".format(num_blood_val))
    print("Number of noblood imgs in validation set: {}".format(int(num_blood_val * factor)))

    print("Number of   blood imgs in training set: {}".format(num_blood_train))
    print("Number of noblood imgs in training set: {}".format(int(num_blood_train * factor)))


def load_dataset(crossval_element, *, input_transform, output_transform, batch_size):
    """Create dataloaders for each crossvalidation item.
    Args:
        crossval_element (dict)                             One step of cross validation
        crossval_element.val_blood (list of strings)        Clip(in DATASET_BLOOD) belongs to validation set if folder contains any keyword in this list.
        crossval_element.val_noblood (list of strings)      Clip(in DATASET_NO_BLOOD) belongs to validation set if folder contains any keyword in this list.
        crossval_element.train_noblood (list of strings)    Hard coded no blood clips. Training set should not contains any clip which simillar to validation set.

    Return:
        dataloaders (dict)                                  Contains two dataloaders for trainings and validation set
        dataloaders.train                                   Training set, loaded by torch.utils.data.DataLoader
        dataloaders.val                                     Validation set, loaded by torch.utils.data.DataLoader
    """

    # list of absolute path tuple to validation and training image, tuple structure: (ori_img, mask_img)
    val_blood_list, train_blood_list = [], []

    # fill them by os.walk
    for fpath, _, fnames in os.walk(DATASET_BLOOD, followlinks=True):
        _, clip_id = os.path.split(fpath)
        fnames = sorted(fnames)

        assert len(fnames) % 2 == 0, "Missing mask image under path: {}".format(fpath)

        if any(key in clip_id for key in crossval_element["val_blood"]):
            val_blood_list += list((os.path.join(fpath, fnames[2*index]), os.path.join(fpath, fnames[2*index+1])) for index in range(int(len(fnames) / 2)))
        else:
            train_blood_list += list((os.path.join(fpath, fnames[2*index]), os.path.join(fpath, fnames[2*index+1])) for index in range(int(len(fnames) / 2)))

    assert all( m == '_mask'.join(os.path.splitext(f)) for (f, m) in train_blood_list), "Some img-mask pair not match."
    assert all( m == '_mask'.join(os.path.splitext(f)) for (f, m) in val_blood_list), "Some img-mask pair not match."

    # list of absolute path tuple to validation and training noblood image, tuple structure: (ori_img, mask_img)
    val_noblood_list, train_noblood_list = [], []

    # fill them by os.walk
    for fpath, _, fnames in os.walk(DATASET_NO_BLOOD, followlinks=True):
        _, clip_id = os.path.split(fpath)

        assert len(fnames) % 2 == 0, "Missing mask image under path: {}".format(fpath)

        if any( key in clip_id for key in crossval_element["val_noblood"]):
            val_noblood_list += list((os.path.join(fpath, sorted(fnames)[2*index]), os.path.join(fpath, sorted(fnames)[2*index+1])) for index in range(int(len(fnames) / 2)))
        else:
            train_noblood_list += list((os.path.join(fpath, sorted(fnames)[2*index]), os.path.join(fpath, sorted(fnames)[2*index+1])) for index in range(int(len(fnames) / 2)))

    assert all( m == '_mask'.join(os.path.splitext(f)) for (f, m) in val_noblood_list), "Some img-mask pair not match."
    assert all( m == '_mask'.join(os.path.splitext(f)) for (f, m) in train_noblood_list), "Some img-mask pair not match."

    # random select follow BLOOD_IMG_PROPORTION
    random.seed(RANDOM_SEED)
    factor = (1 - BLOOD_IMG_PROPORTION) / BLOOD_IMG_PROPORTION
    val_noblood_list = random.sample(val_noblood_list, k=int(len(val_blood_list) * factor))
    train_noblood_list = random.sample(train_noblood_list, k=int(len(train_blood_list) * factor))

    # combine blood and noblood images
    val_list = val_blood_list + val_noblood_list
    train_list = train_blood_list + train_noblood_list

    crossval_obj = {'train': train_list, 'val': val_list}    
    
    image_datasets = {x: SegmentationImageFolder(crossval_obj[x], input_transform=input_transform, output_transform=output_transform, mask_channel=1) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes
    

############################################################
####################        model       ####################
############################################################
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


class ConfusionMatrix(nn.Module):

    def __init__(self):
        super(ConfusionMatrix, self).__init__()
        self.smooth = 1.0
        self.threshold = torch.Tensor([0.5]).to(device)

    def forward(self, y_pred, y_true):
        y_pred = (y_pred.view(-1) > self.threshold).float()
        y_true = y_true.view(-1)

        assert y_pred.size() == y_true.size()

        TP = (y_pred * y_true).sum()
        FP = (y_pred * (1 - y_true)).sum()
        FN = ((1 - y_pred) * y_true).sum()
        TN = ((1 - y_pred) * (1 - y_true)).sum()

        return torch.Tensor([[TP, FP], [FN, TN]]).to(torch.long)
        

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


############################################################
####################  training function ####################
############################################################
def train_model(model, *, dataset_sizes, dataloaders, title, criterion, optimizer, scheduler, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    loest_loss = 1.0

    progress = {
        'train': {},
        'val': {}
    }

    confusion_matrix_analyser = ConfusionMatrix()
    confusion_matrix_analyser.to(device)

    start_time = datetime.today()

    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{} starts at {}'.format(epoch + 1, num_epochs, start_time.strftime('%Y.%m.%d, %H:%M:%S')))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            confusion_matrix = torch.zeros([2,2], dtype=torch.long)

            # for each batch
            for inputs, mask, fname in dataloaders[phase]:

                inputs = inputs.to(device)
                mask = mask.to(device)

                """
                # check if mask and img are transformed identically
                img_folder_path = os.path.join(SAVE_ROOT, DATE_INFO, '123')
                if not os.path.isdir(img_folder_path):
                    os.mkdir(img_folder_path)
                mname = '_mask'.join(os.path.splitext(fname[0]))
                torchvision.utils.save_image(inputs, os.path.join(img_folder_path, fname[0]))
                torchvision.utils.save_image(mask, os.path.join(img_folder_path, mname))
                """

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, mask)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                confusion_matrix += confusion_matrix_analyser(outputs.to(device), mask)

                # statistics
                running_loss += loss.item() * inputs.size(0)

            # log
            epoch_loss = running_loss / dataset_sizes[phase]
            val_dict = val_analyse(confusion_matrix)
            val_dict['loss'] = epoch_loss
            for key in val_dict:
                value = val_dict[key]
                progress[phase][key] = progress[phase][key] if key in progress[phase] else []
                progress[phase][key].append(value)

            print(
                '{:5} || loss: {:.3f} percision: {:.3f} recall: {:.3f} accuracy: {:.3f} specificity: {:.3f} f1: {:.3f}'.format(
                    phase, 
                    epoch_loss, 
                    val_dict["percision"] if val_dict["percision"] else "None", 
                    val_dict["recall"] if val_dict["recall"] else "None", 
                    val_dict["accuracy"] if val_dict["accuracy"] else "None", 
                    val_dict["specificity"] if val_dict["specificity"] else "None",
                    val_dict["f1"] if val_dict["f1"] else "None"
                ), 
                '||', 
                ' TP:{:10d} FP:{:10d} FN:{:10d} TN:{:10d}'.format(
                    val_dict["TP"],
                    val_dict["FP"], 
                    val_dict["FN"], 
                    val_dict["TN"]
                )
            )

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            if phase == 'val':

                if not os.path.isdir(os.path.join(SAVE_ROOT, DATE_INFO)):
                    os.mkdir(os.path.join(SAVE_ROOT, DATE_INFO))

                if epoch_loss < loest_loss:
                    loest_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(SAVE_ROOT, DATE_INFO, '{}.pt'.format(title + '_LL')))
                    progress['loest_loss_save'] = {
                        'num_epoch': epoch,
                        'validation_info': val_dict
                    }

        end_time = datetime.today()
        print('{}.{} seconds tooked'.format(
            (end_time - start_time).seconds,
            (end_time - start_time).microseconds
        ))

        start_time = datetime.today()

    # save after last epoch
    torch.save(model.state_dict(), os.path.join(SAVE_ROOT, DATE_INFO, '{}.pt'.format(title + '_LE')))
    progress['last_epoch_save'] = {
        'num_epoch': epoch,
        'validation_info': val_dict
    }

    # save evaluation svg
    show_plot_l = ['percision', 'recall', 'accuracy', 'specificity', 'f1', 'loss']
    color_l = ['b', 'r', 'g', 'c', 'm', 'y']
    fig = plt.figure(figsize=(24,9))
    fig.suptitle(title, fontsize="x-large")
    for index, key in enumerate(show_plot_l):
        plt.subplot(2, 3, index+1)
        plt.title(key)
        for phase in ['train', 'val']:
            color = (color_l[index] + '-') if phase == 'val' else '--'
            y = progress[phase][key]
            x = range(len(y))
            plt.gca().set_ylim(bottom=0, top=1)
            plt.plot(x,y, color)
    svg_path = os.path.join(SAVE_ROOT, DATE_INFO, "{}.svg".format(title))
    plt.savefig(svg_path)

    # save evaluation json 
    json_path = os.path.join(SAVE_ROOT, DATE_INFO, '{}.json'.format(title))
    with open(json_path, 'w') as outfile:
        outfile.seek(0)
        outfile.truncate(0)
        json.dump(progress, outfile)

    """
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
    """

def val_analyse(confusion_matrix):
    """ Calculate TP, FP, FN, TN, percision, recall, accuracy, specificity, f1

    Args:
        confusion_matrix (torch.Tensor):    tensor with shape [2,2] and type torch.int32

    Returns:
        val_dict (dict):                         Packed as dict, because Tian doesnt like very long list
        val_dict["TP"] (long)                    Number of pixel, pred=1 and mask=1
        val_dict["FP"] (long)                    Number of pixel, pred=1 and mask=0
        val_dict["FN"] (long)                    Number of pixel, pred=0 and mask=1
        val_dict["TN"] (long)                    Number of pixel, pred=0 and mask=0
        val_dict["percision"] (long)             Number of correct predicted pixel divided by the number of all pixel, which predicted as True.
        val_dict["recall"] (long)                Number of correct predicted pixel divided by the number of all pixel, which should be predicted as True.
        val_dict["accuracy"] (long)              
        val_dict["specificity"] (long)           
        val_dict["f1"] (long)                    
    """
    val_dict = {}

    basic = ["TP", "FP", "FN", "TN"]
    val_dict.update(dict((basic[index], confusion_matrix.view(-1).tolist()[index]) for index in range(len(basic))))

    def weird_division(x, y):
        return x / y if y else None

    val_dict["percision"] = weird_division(val_dict["TP"], (val_dict["TP"] + val_dict["FP"]))
    val_dict["recall"] = weird_division(val_dict["TP"], (val_dict["TP"] + val_dict["FN"]))
    val_dict["accuracy"] = weird_division((val_dict["TP"] + val_dict["TN"]), (val_dict["TP"] + val_dict["FP"] + val_dict["FN"] + val_dict["TN"]))
    val_dict["specificity"] = weird_division(val_dict["TN"], (val_dict["TN"] + val_dict["FP"]))
    if val_dict["percision"] and val_dict["recall"]:
        val_dict["f1"] = weird_division(2 * val_dict["percision"] * val_dict["recall"], (val_dict["percision"] + val_dict["recall"]))
    else:
        val_dict["f1"] = None
    return val_dict


def do_segmentation(model, *, dataset_sizes, dataloaders, title, maskRGB, predRGB):
    img_folder_path = os.path.join(SAVE_ROOT, DATE_INFO, title)
    threshold = torch.Tensor([0.5]).to(device)
    if not os.path.isdir(img_folder_path):
        os.mkdir(img_folder_path)

    for inputs, mask, fname in dataloaders['val']:
        inputs = inputs.to(device)
        mask = (mask.to(device) > threshold).float()
        outputs = (model(inputs) > threshold).float() 

        # edge detection
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).view(1,1,3,3).float().to(device)
        outputs_edged = outputs * (1. - (torch.nn.functional.conv2d(outputs, kernel, padding=1) == 4.).float())
        mask_edged = mask * (1. - (torch.nn.functional.conv2d(mask, kernel, padding=1) == 4.).float())

        # change color
        outputs_edged_colored = torch.cat([outputs_edged * predRGB[0], outputs_edged * predRGB[1], outputs_edged * predRGB[2]], dim=1)
        mask_edged_colored = torch.cat([mask_edged * maskRGB[0], mask_edged * maskRGB[1], mask_edged * maskRGB[2]], dim=1)

        # draw edges
        inputs = inputs * (1. - torch.cat([mask_edged, mask_edged, mask_edged], dim=1)) + mask_edged_colored
        inputs = inputs * (1. - torch.cat([outputs_edged, outputs_edged, outputs_edged], dim=1)) + outputs_edged_colored

        # output
        torchvision.utils.save_image(inputs, os.path.join(img_folder_path, fname[0]))



############################################################
####################     run (train)    ####################
############################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if not os.path.isdir(os.path.join(SAVE_ROOT, DATE_INFO)):
    os.mkdir(os.path.join(SAVE_ROOT, DATE_INFO))

training_inf = {
    'model': DESCRIPTION,
    'DATE_INFO': DATE_INFO,
    'LR': LR,
    'BATCH_SIZE': BATCH_SIZE,
    'NUM_EPOCHS': NUM_EPOCHS,
    'DATA_TRANSFORMS': str(DATA_TRANSFORMS)
}
json_path = os.path.join(SAVE_ROOT, DATE_INFO, 'training_inf.json')
with open(json_path, 'w') as outfile:
    outfile.seek(0)
    outfile.truncate(0)
    json.dump(training_inf, outfile)

# for each validation step
for crossval_element in CROSSVAL_LIST:

    print("###############################################")
    print("CROSS VALIDATION STEP: " + crossval_element['title'])
    print()

    report_crossval_element(crossval_element)
    
    dataloaders, dataset_sizes = load_dataset(crossval_element, input_transform=DATA_TRANSFORMS, output_transform=DATA_TRANSFORMS, batch_size=BATCH_SIZE)
    
    unet = UNet(in_channels=3, out_channels=1)
    unet.to(device)

    criterion = DiceLoss()
    criterion.to(device)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(unet.parameters(), lr=LR, momentum=0.9)
    optimizer_ft = optim.Adam(unet.parameters(), lr=LR)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.25)
    exp_lr_scheduler = None

    # train
    train_model(
        unet, 
        dataset_sizes = dataset_sizes,
        dataloaders = dataloaders, 
        title = crossval_element['title'], 
        criterion = criterion, 
        optimizer = optimizer_ft, 
        scheduler = exp_lr_scheduler, 
        num_epochs = NUM_EPOCHS
    )

    # reload data for validation
    dataloaders, dataset_sizes = load_dataset(crossval_element, input_transform=DATA_TRANSFORMS_NOFLIP, output_transform=DATA_TRANSFORMS_NOFLIP, batch_size=1)

    # LL: loest loss, LE: last epoch
    for padding in ['_LL', '_LE']:
        fresh_unet = UNet(in_channels=3, out_channels=1)
        fresh_unet.to(device)
        state_dict = torch.load(os.path.join(SAVE_ROOT, DATE_INFO, '{}.pt'.format(crossval_element['title'] + padding)))
        fresh_unet.load_state_dict(state_dict)
        fresh_unet.eval()

        do_segmentation(
            fresh_unet, 
            dataset_sizes = dataset_sizes,
            dataloaders = dataloaders,
            title = crossval_element['title'] + padding,
            maskRGB = [0, 255, 0], 
            predRGB = [255, 255, 0]
        )
    
