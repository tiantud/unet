############################################################
####################       import       ####################
############################################################
import os
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

from Segmentationdatasets_Restnet import SegmentationImageFolder


############################################################
####################  global variables  ####################
############################################################

# start date
DESCRIPTION = 'fcn_restnet50'
DATE_INFO = datetime.today().strftime('%Y%m%d_%H%M%S')

# set seed
RANDOM_SEED = 2020

# training param
LR = 0.0001
BATCH_SIZE = 8
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
        "title": "val_n2",
        "val_blood": ["SER01"],
        "val_noblood": ["SER16_NO"],
        "train_noblood": ["SER15_NO", "SER18_NO"]
    },
    {
        "title": "val_n1",
        "val_blood": ["SER02"],
        "val_noblood": ["SER18_NO"],
        "train_noblood": ["SER15_NO", "SER16_NO"]
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

DATA_INPUT_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

DATA_OUTPUT_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

DATA_TRANSFORMS_NOFLIP = transforms.Compose([
    transforms.Resize((256, 256)),
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

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=0, max=1)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        assert y_pred.size() == y_true.size()

        TP = (y_pred * y_true).sum()
        FP = (y_pred * (1 - y_true)).sum()
        FN = ((1 - y_pred) * y_true).sum()
        TN = ((1 - y_pred) * (1 - y_true)).sum()

        return torch.Tensor([[TP, FP], [FN, TN]]).to(torch.long)


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
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

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
            for inputs, _, mask, fname in dataloaders[phase]:

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
                    outputs = model(inputs)['out'].permute(1,0,2,3)
                    outputs = (outputs[1] - outputs[0]).unsqueeze(0).permute(1,0,2,3)

                    classifier = nn.Sigmoid()
                    outputs = classifier(outputs)
                    # ['out'].argmax(dim=1, keepdim=True).float()
                    
                    loss = criterion(outputs, mask)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                confusion_matrix += confusion_matrix_analyser(outputs, mask)

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

    for inputs, inputs_no_normalize, mask, fname in dataloaders['val']:
        inputs = inputs.to(device)
        mask = (mask.to(device) > threshold).float()
        outputs = model(inputs)['out'].permute(1,0,2,3)
        outputs = (outputs[1] - outputs[0]).unsqueeze(0).permute(1,0,2,3)

        classifier = nn.Sigmoid()
        outputs = (classifier(outputs) > threshold).float() 

        mname = '_mask'.join(os.path.splitext(fname[0]))
        oname = '_output'.join(os.path.splitext(fname[0]))

        # edge detection
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).view(1,1,3,3).float().to(device)
        outputs_edged = outputs * (1. - (torch.nn.functional.conv2d(outputs, kernel, padding=1) == 4.).float())
        mask_edged = mask * (1. - (torch.nn.functional.conv2d(mask, kernel, padding=1) == 4.).float())

        # change color
        outputs_edged_colored = torch.cat([outputs_edged * predRGB[0], outputs_edged * predRGB[1], outputs_edged * predRGB[2]], dim=1)
        mask_edged_colored = torch.cat([mask_edged * maskRGB[0], mask_edged * maskRGB[1], mask_edged * maskRGB[2]], dim=1)

        # draw edges
        inputs_draw = inputs_no_normalize.to(device)
        inputs_draw = inputs_draw * (1. - torch.cat([mask_edged, mask_edged, mask_edged], dim=1)) + mask_edged_colored
        inputs_draw = inputs_draw * (1. - torch.cat([outputs_edged, outputs_edged, outputs_edged], dim=1)) + outputs_edged_colored

        # output
        torchvision.utils.save_image(inputs_draw, os.path.join(img_folder_path, fname[0]))



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
    'DATA_INPUT_TRANSFORMS': str(DATA_INPUT_TRANSFORMS),
    'DATA_OUTPUT_TRANSFORMS': str(DATA_OUTPUT_TRANSFORMS)
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
    
    dataloaders, dataset_sizes = load_dataset(crossval_element, input_transform=DATA_INPUT_TRANSFORMS, output_transform=DATA_OUTPUT_TRANSFORMS, batch_size=BATCH_SIZE)
    
    model = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
    model.to(device)

    criterion = DiceLoss()
    criterion.to(device)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    optimizer_ft = optim.Adam(model.parameters(), lr=LR)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.25)
    exp_lr_scheduler = None

    # train
    train_model(
        model, 
        dataset_sizes = dataset_sizes,
        dataloaders = dataloaders, 
        title = crossval_element['title'], 
        criterion = criterion, 
        optimizer = optimizer_ft, 
        scheduler = exp_lr_scheduler, 
        num_epochs = NUM_EPOCHS
    )

    # reload data for validation
    dataloaders, dataset_sizes = load_dataset(crossval_element, input_transform=DATA_INPUT_TRANSFORMS, output_transform=DATA_OUTPUT_TRANSFORMS, batch_size=1)

    # LL: loest loss, LE: last epoch
    for padding in ['_LL', '_LE']:
        fresh_model = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
        fresh_model.to(device)
        state_dict = torch.load(os.path.join(SAVE_ROOT, DATE_INFO, '{}.pt'.format(crossval_element['title'] + padding)))
        fresh_model.load_state_dict(state_dict)
        fresh_model.eval()

        do_segmentation(
            fresh_model, 
            dataset_sizes = dataset_sizes,
            dataloaders = dataloaders,
            title = crossval_element['title'] + padding,
            maskRGB = [0, 255, 0], 
            predRGB = [255, 255, 0]
        )
    
