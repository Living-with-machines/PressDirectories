from fastai.vision.learner import unet_learner
from torch import threshold
from tools.helpers import range_to_pagenumbers
from pathlib import Path, PosixPath
import shutil
import numpy as np
from tqdm.notebook import tqdm
import cv2
from typing import Union
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from PIL import Image
from sklearn.linear_model import LinearRegression
import cv2

create_img_path = lambda page, year, DATA: DATA / f'MPD_{year}' / 'Images' / f"MPD_{year}_{str(page).zfill(3)}.tif"

to_pages = lambda editions, data: [create_img_path(p, year, data) for year,pages in editions.items() for p in pages]

# -------- image processing --------
# moving and converting images

class ImageClass(object):
    """Class for reading and converting images to .PNG

    Arguments:
        image_path (PosixPath): path to image
        out_folder (PosixPath): folder for storing converted image
                                if not set, parent / "Images_converted" is used
    """
    def __init__(self,image_path: PosixPath,
                out_folder: PosixPath = Path("./to_annotate")):
        self.image_path = image_path
        self.parent = image_path.parent.parent
        self.name = image_path.name
        if not out_folder:
            self.out_folder = self.parent/ "Images_converted"
        else:
            self.out_folder = out_folder
        self.out_folder.mkdir(exist_ok=True)
        
    def convert(self):
        """convert image, uses the main functionalities attached to the class
        namely loading the image, converting to grayscale and saving the images
        """
        self.load_image()
        self.to_grayscale()
        self.save()
        
    def load_image(self):
        """load image with cv2"""
        self.image = cv2.imread(str(self.image_path))
    
    def to_grayscale(self):
        """convert image to grayscale"""
        self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
    def save(self,ext='png'):
        """save image as .PNG"""
        cv2.imwrite(str(self.out_folder / (self.image_path.stem  + f'.{ext}')), self.grayscale)

def move_selected_from_blobstore_image(path: PosixPath, 
                                    root_folder: PosixPath):
    """move _and_ convert a selected image from blobstore to specified root folder
    this is need because we don't want to write to the blobstorage
    Arguments:
        path (PosixPath): path to image
        root_folder (PosixPath): move image to root folder / edition_id / Images
    """
    if not path.is_file(): 
        print(f"Error, file {path.stem} doesn't exist.")
        return path.stem  
    
    # edition of Mitchell
    edition = path.parent.parent.name
    edition_folder = root_folder / edition
    edition_folder.mkdir(exist_ok=True)
    images_folder = edition_folder / "Images"
    images_folder.mkdir(exist_ok=True)
    # check if file exists
    if (images_folder / (path.stem + ".png")).is_file():
        print(f"skipping {path.stem}")
    else:
        try:
            # try to convert image
            ImageClass(path,out_folder=images_folder).convert()
            return True
        except Exception as e:
            print(f'Error {e} while converting ile {path.stem}')

# -------- image classification: data prep --------
# prepare mask from annotated data


def draw_polygons(json_file: dict,
                orig_path: PosixPath=Path('./to_annotate'), 
                out_path: PosixPath=Path('columns'),
                codes: dict=None,
                fill=True):
    """function that draws a polygon give coordinates
    it operates on the folder level
    this function creates masks for training the unet
    
    Arguments:
        json_file (dict): json output of VIA annotation tool
        orig_path (PosixPath): folder that contains images annotated in VIA 
        out_path (PoxixPath): folder to save labels (masks) and corresponding images 
        codes (dict): a mapping from str to int, which maps labels to and integer
                    the serves as id for a specific segment, i.e. left column = 2
    """
    
    def draw_polygon(img_metadata: dict):
        """draw polygon for a specific image move mask and image
        for a folder structure that is expected by the SegmentationItemList
        function (fastAI)
            root folder (out_path)
                |_ images
                |_ labels
                
        Argument:
            img_metadata (dict): metadata of images as provided by VIA JSON export
                                contains information about the file as well as the 
                                annotations (under regions > shape_attribtutes)
            
        """

        # get name
        name = img_metadata['filename']
        
        height, width, channels = cv2.imread(str(orig_path / name)).shape
        # create black image of the size equal to the input image
        mask = np.zeros((height, width), dtype = "uint8")
        # get annotations
        regions = img_metadata['regions']
        
        # if not annotations, return False
        if not regions: return False
        
        # move original image
        shutil.copy(orig_path / name, img_save / name)
        
        # for each region, create a mask and add it the image
        for region in regions:
            coords =  list(zip(region['shape_attributes']['all_points_x'],
                        region['shape_attributes']['all_points_y']))

            if fill:
                mask = cv2.fillPoly(mask, [np.array(coords)], codes[region['region_attributes']['column']])
            else:
                mask = cv2.polylines(mask, 
                                    [np.array(coords)], 
                                    color=codes[region['region_attributes']['column']],
                                    isClosed=True,
                                    thickness=25)
        # save mask in the labels folder
        cv2.imwrite(str(labels_save / name), mask)
        
        return True
    
    # if no mapping is provided use the avialable attributes 
    if codes is None:
        codes = {k:(i+1) for i,(k,v) in enumerate(json_file['_via_attributes']['region']['column']['options'].items())}#['options']
    
    # prepare folders
    out_path.mkdir(exist_ok=True)
    img_save = (out_path / 'images')
    img_save.mkdir(exist_ok=True)
    labels_save = (out_path / 'labels')
    labels_save.mkdir(exist_ok=True)
    
    for img_key,img_metadata in tqdm(json_file['_via_img_metadata'].items()):
        draw_polygon(img_metadata)

def draw_rectangles(json_file: dict,
                orig_path: PosixPath=Path('./to_annotate_capitals'), 
                out_path: PosixPath=Path('capitals'),
                codes: dict=None):
    """function that draws a polygon give coordinates
    it operates on the folder level
    this function creates masks for training the unet
    
    Arguments:
        json_file (dict): json output of VIA annotation tool
        orig_path (PosixPath): folder that contains images annotated in VIA 
        out_path (PoxixPath): folder to save labels (masks) and corresponding images 
        codes (dict): a mapping from str to int, which maps labels to and integer
                    the serves as id for a specific segment, i.e. left column = 2
    """
    
    def draw_rectangle(img_metadata: dict):
        """draw rectangle for a specific image move mask and image
        for a folder structure that is expected by the SegmentationItemList
        function (fastAI)
            root folder (out_path)
                |_ images
                |_ labels
                
        Argument:
            img_metadata (dict): metadata of images as provided by VIA JSON export
                                contains information about the file as well as the 
                                annotations (under regions > shape_attribtutes)
            
        """

        # get name
        name = img_metadata['filename']
        
        height, width, channels = cv2.imread(str(orig_path / name)).shape
        # create black image of the size equal to the input image
        mask = np.zeros((height, width), dtype = "uint8")
        # get annotations
        regions = img_metadata['regions']
        
        # if not annotations, return False
        if not regions: return False
        
        # move original image
        shutil.copy(orig_path / name, img_save / name)
        
        # for each region, create a mask and add it the image
        for region in regions:
            x =  region['shape_attributes']['x']; y =  region['shape_attributes']['y']
            w =  region['shape_attributes']['width']; h =  region['shape_attributes']['height']
            #print([(x,y),(x+w,y+h)])
            mask = cv2.rectangle(mask, (x,y),(x+w,y+h), codes['capital'],-1) # clunky but whatever
        # save mask in the labels folder
        cv2.imwrite(str(labels_save / name), mask)
        
        return True
    # if no mapping is provided use the avialable attributes 
    if codes is None:
        codes = {k:(i+1) for i,(k,v) in enumerate(json_file['_via_attributes']['region']['capital']['options'].items())}#['options']
    
    # prepare folders
    out_path.mkdir(exist_ok=True)
    img_save = (out_path / 'images')
    img_save.mkdir(exist_ok=True)
    labels_save = (out_path / 'labels')
    labels_save.mkdir(exist_ok=True)


    for img_key,img_metadata in tqdm(json_file['_via_img_metadata'].items()):
        draw_rectangle(img_metadata)

# training a model on masked data


#def accuracy(input, 
#            target,
#            void_code: int=999):
#    """computes pixel level accuracy"""
#    target = target.squeeze(1)
#    mask = target != void_code
#    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# -------- image classification: train a model --------
# train or continue training a model

def train_model(path: str,
                split_by_pct: float=.15,
                bs:int=1,
                wd:float=1e-2,
                epochs: int=10,
                size: tuple=(1100,1100),
                learn = None,
                root: PosixPath=Path('.'),
                export_path: PosixPath=Path('')
                ):
    """
    function that contains all training stepss

    Arguments:
        path (str): folder name where labels and images are s
        split_by_pct (float): percentage for train / dev split
        bs (int): batch size
        wd (float): ?
        epochs (int): number of epochs needed for training
        size (tuple): convert input image to size of (width, height) tuple
        learn (unet_learner): of a model is given, we continue training, 
                            otherwise initialize a new model
        root: (PosixPath): root path, i.e. working directory to contains path folder
        export_path (PosixPath): export model to export_path folder
    """ 
    get_y_fn = lambda x: path_lbl/f'{x.stem}.png'

    path_lbl = root / path / 'labels'
    path_img = root / path / 'images'
    codes = np.array(['page','column','left_col','righ_col'])

    src = (SegmentationItemList.from_folder(path_img)
        .split_by_rand_pct(split_by_pct)
        .label_from_func(get_y_fn, classes=codes))

    data = (src.transform(size=size, tfm_y=True)  # (900,600)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

    if learn is None:
        learn = unet_learner(data, 
                        models.resnet34, 
                        metrics=accuracy, 
                        wd=wd)
    print('finding learning rate')
    lr_find(learn)
    learn.recorder.plot()
    lr = float(input('enter learning rate: '))
    
    learn.fit_one_cycle(epochs, slice(lr), pct_start=0.8)

    export_path.mkdir(exist_ok=True)
    model_path = export_path / 'export.pkl'
    learn.export(model_path)
    return learn

# -------- image postprocessing --------
# extract columns


def return_mask(path: PosixPath,
                learn) -> np.array:
    """return predictions for an image
    these prediction are used to recreate
    masks and extract target entities such as columns
    
    Arguments:
        path (PosixPath): path to image
        learn (unet_learner): trained unet model
    """
    # open image and predict pixel
    x = open_image(path)
    #print(path)
    #x = Image.open(path)
    predictions = learn.predict(x)
    
    # open image with pillow
    image = Image.open(path)
    # convert to array
    image_array = np.array(image)
    height, width = image_array.shape
    
    # get the mask tenser
    mask = predictions[1]
    # remove dimension convert to uint8
    mask_image = Image.fromarray(np.array(mask.squeeze(0)).astype(np.uint8))
    # resize mask to size of the original image
    mask_image_resized = mask_image.resize((width,height))
    # convert resized mask to array
    mask_image_resized_array = np.array(mask_image_resized)
    
    return mask_image_resized_array

def replace_after_last(row: np.array,
                        code: int,
                        replacement: int) -> np.array: 
    """replace "code (int)" with "replacement (int)" in array
    after the last occurence of "code (int)"

    Arguments:
        row (np.array): array with integers
        code (int): target code which we want to replace
        replacement (int): replace code with replacements integer
    """
    locations = np.where(row==code)
    if len(locations[0]) > 0:
        last = locations[0][-1]
        row[last:] = replacement
    return row

def replace_before_first(row: np.array,
                        code: int,
                        replacement: int) -> np.array:
    """replace "code (int)" with "replacement (int)" in array
    before the first occurence of "code (int)"

    Arguments:
        row (np.array): array with integers
        code (int): target code which we want to replace
        replacement (int): replace code with replacements integer
    """
    locations = np.where(row==code)
    if len(locations[0]) > 0:
        first = locations[0][0]
        row[:first] = replacement
    return row

def get_first(row: np.array,
            code:int) -> np.array:
    """get first index position of target code in array
    if code not found return the last index

    Arguments:
        row (np.array): array with integer values
        code (int): target code
    """
    locations = np.where(row==code)
    if len(locations[0]) > 0:
        first = locations[0][0]
    else:
        first = len(row)
    return first

def get_last(row: np.array,
            code:int) -> np.array:
    """get last index position of target code in array
    if code not found return 0 (first index)

    Arguments:
        row (np.array): array with integer values
        code (int): target code
    """
    locations = np.where(row==code)
    if len(locations[0]) > 0:
        last = locations[0][-1]
    else:
        last = 0
    return last

def blank(row: np.array,
        code: int,
        threshold:float=.25) -> np.array:
    """strike out rows in image
    based on the ratio of a target code 
    to the total number of pixels in row
    
    Arguments:
        row (np.array)

    """
    locations = np.where(row==code)
    # if proportion of target code pixels
    # is fewer than the threshold
    # put everything to black
    if float(len(locations[0])) / float(len(row)) < threshold:
        row = np.zeros(len(row))
    return row

def smooth_coefficient(coeff,smooth_facor=1e-5):
    if -.05 < coeff < .05:
        return coeff
    return coeff*smooth_facor

def get_coefficients(side,content):
    y = side[content]
    x = np.array(range(len(y))).reshape(-1,1)
    reg = LinearRegression().fit(x, y)
    return smooth_coefficient(reg.coef_[0]),reg.intercept_

def poly_to_bb(poly_coord):
    x_coords = [i[0] for i in poly_coord]
    y_coords = [i[1] for i in poly_coord]
    x = min(x_coords)
    w = max(x_coords) - min(x_coords)
    y = min(y_coords)
    h = max(y_coords) - min(y_coords)
    return x,y,w,h

def filter_values_by_std(array):
    mean = np.mean(array)
    std = np.std(array)
    z = (array - mean) / std
    return array[np.where(z <= 1.0 )[0]]

def get_page(image_path,mask_array):
    image_array = np.array(Image.open(image_path))
    image_array[mask_array != 1] = 125
    return image_array

def get_columns(image_path,mask_array,buffer=12): # ,split=True
    print(np.unique(mask_array))
    
    def get_column(col,col_mask,code):
        side = np.apply_along_axis(get_first,1,col_mask,code=code)


        content = np.where(side < 300)
        print(side)
        print(content)

        coeff, bias = get_coefficients(side,content)
        
        l_start = int(bias - content[0][0]*coeff)
        l_end = int(bias+((col_mask.shape[0]-content[0][0])*coeff))
        l_line = [(l_start+int((l_start+i)*coeff),i) for i in range(col_mask.shape[0])]
        l_line.extend([(i+1,j) for i,j in l_line])
        side = np.apply_along_axis(get_last,1,col_mask,code=code)
        content = np.where(side > col_mask.shape[1]-600)

        coeff, bias = get_coefficients(side,content)
        r_start = int(bias - content[0][0]*coeff)
        r_end = int(bias+((col_mask.shape[0]-content[0][0])*coeff))
        r_line = [(r_start+int((r_start+i)*coeff),i) for i in range(col_mask.shape[0])]

        col_mask_t = col_mask.T
        side = np.apply_along_axis(get_first,1,col_mask_t,code=code)
        
        content = np.where(side < 1000)
        coeff, bias= get_coefficients(side,content)
        bias-=buffer
        t_start = int(bias - content[0][0]*coeff)
        t_line = [(i,t_start+int((t_start+i)*coeff)) for i in range(col_mask.shape[1])]
        t_line.extend([(i,j+1) for i,j in t_line])
        t_end = int(bias+((col_mask_t.shape[0]-content[0][0])*coeff))
        
        side = np.apply_along_axis(get_last,1,col_mask_t,code=code)
        
        content = np.where(side > col_mask_t.shape[1] - 1000)
        
        coeff, bias = get_coefficients(side,content)
        bias+=buffer    
        b_start = int(bias - content[0][0]*coeff)
        b_line = [(i,int(b_start+(i*coeff))) for i in range(col_mask.shape[1])]
        b_end = int(bias+ ((col_mask_t.shape[0]-content[0][0])*coeff))
        try:
            lt = list(set(t_line).intersection(l_line))
            if lt:
                lt = lt[0]
            else:
                lt = (min(0,l_start),min(col_mask.shape[0],t_end))

            rt = list(set(t_line).intersection(r_line))
            if rt:
                rt = rt[0]
            else:
                rt = (min(col_mask.shape[1],r_end),min(col_mask.shape[0],t_end))

            lb = list(set(b_line).intersection(l_line))
            if lb:
                lb = lb[0]
            else:
                lb = (min(0,l_end),min(col_mask.shape[0],b_end))
            
            rb = list(set(b_line).intersection(r_line))
            if rb:
                rb=rb[0]
            else:
                rb = (min(col_mask.shape[1],r_end),min(col_mask.shape[0],b_end))
            #print(rb)

        except:
            print(image_path)
            print("Error returning image for debugging ", code)
            
            cv2.line(col,(l_start,0),(l_end,col.shape[0]), 255 ,10)
            cv2.line(col,(r_start,0),(r_end,col.shape[0]), 255 ,10)
            cv2.line(col,(0,b_start),(col.shape[1],b_end), 255 ,10)
            cv2.line(col,(0,t_start),(col.shape[1],t_end), 255 ,10)
            col[col_mask == code] = 125 
            return col
        
        new_mask = np.zeros((col.shape[0],col.shape[1]))
        
        cv2.fillPoly(new_mask,np.array([np.array([rt,rb,lb,lt])]),color=code)
        col[new_mask!= code] = 200
        x,y,w,h = poly_to_bb([rt,rb,lb,lt])
        col = col[max(0,y):min(y+h,col.shape[0]),max(0,x):min(x+w,col.shape[1])]

        return col

    # average pixel over all the columns

    #if split:
    mean_pixel_x = np.apply_along_axis(np.mean,0,mask_array)
    # compute the difference between consecative means
    # mean[i+1] - mean[i]
    diff = np.diff(mean_pixel_x)
    # the largest difference, is where we split
    # where it jumps from mainly 2 to mainly 3 code
    split_middle = np.argmin(diff[1000:-1000])
    # pd.Series(diff).plot()
    # add a bit of buffer
    left_col_mask = mask_array[:,:(1000+split_middle+buffer)]
    right_col_mask = mask_array[:,1000+split_middle-buffer:]
    #def column(col_mask):
    # left column left side
    image_array = np.array(Image.open(image_path))
    left_col = image_array[:, :(left_col_mask.shape[1])].copy()
    right_col = image_array[:, -right_col_mask.shape[1]:].copy()

    left_col = get_column(left_col,left_col_mask,1)
    right_col = get_column(right_col,right_col_mask,2)
    return left_col, right_col
    #else:
    #    image_array = np.array(Image.open(image_path))
    #    return get_column(image_array,mask_array,1)


check_year = lambda path: int(path.stem.split("_")[1])

def save_columns(image_paths: list, 
                model,
                ignore_files:list=[],
                split=True):
    """function that accepts a list of paths to images,
    extracts and saves the columns. input is assumed to follow
    the way we currently store MDP data, processed images
    are stores in Image_processed folder, each file gets 
    _left or _right suffix.

    Arguments:
        image_paths (list): list with paths to image files
        model (unet): trained unet model
        split (bool): if True split into right left, otherwise single page

    Returns:
        save the columns extracted from the image
    """
    for p in tqdm(image_paths):
        #print(p)
        if p in ignore_files: continue
        save_to = p.parent.parent / 'Image_processed'
        save_to.mkdir(exist_ok=True)

        path_left_col = save_to / f"{p.stem}_left.png"
        path_right_col = save_to / f"{p.stem}_right.png"

        left_col = path_left_col.is_file() 
        right_col = path_right_col.is_file()

        # if both columns exist, continue to next
        if left_col and right_col:
            #print(f"Skipping {p.stem}")
            continue
        

        # get predictions
        mask_array = return_mask(p,model)

        # extract columns, if error, print
        try:
            if split:
                l,r = get_columns(p,mask_array,buffer=25) # ,split=split
            else:
                page = get_page(p,mask_array)
        except Exception as e:
            print(f"[WARNING] Error {e} for file {p.stem}")
            continue
        
        # check if left or right columns exists
        if split:
            if not left_col:
                try:
                    Image.fromarray(l).save(path_left_col)
                except Exception as e:
                    print(f"[WARNING] Error {e} for file {p.stem}")
            if not right_col:
                try:
                    Image.fromarray(r).save( path_right_col)
                except Exception as e:
                    print(f"[WARNING] Error {e} for file {p.stem}")
        else:
            path_page = save_to / f"{p.stem}.png"
            #try:
            Image.fromarray(page).save(path_page)
            #except Exception as e:
            #    print(f"[WARNING] Error {e} for file {p.stem}")


# -------- image postprocessing --------
# extract and adjust capitals

def extract_and_adjust_capitals(path: PosixPath,learn) -> Image:
    """function that detects capitals, resizes them,
    and puts the resized capital back in the image

    Arguments:
        path (PosixPath): path to image
        learn (unet): unet model
    """
    
    def get_offsets(pos_idx:list) -> list:
        """helper function that detects boundaries for capitals
        Arg
        """
        i = 0
        # save coordinates
        caps = []
        # initialize empty image
        cap = []
    
        while i < len(pos_idx)-1:
            # if there is a gap of more than five pixel,
            # save existing coordinates, add them to caps
            # and open a new empty list
            if pos_idx[i+1] - pos_idx[i] > 5:
                caps.append(cap)
                cap = []
        
            else:
                # add new coordinate
                cap.append(pos_idx[i])
            i+=1
            # close existing cap at the end of the image
            if (i == len(pos_idx)-1) and cap:
                caps.append(cap)
        # get the first and last coordinates
        return [[min(c),max(c)] for c in caps]
    
    # open image and predict pixel
    x = open_image(path)
    predictions = learn.predict(x)
    
    # open image with pillow
    image = Image.open(path)
    # convert to array
    image_array = np.array(image)
    height, width = image_array.shape
    
    # get the mask tenser
    mask = predictions[1]
    # remove dimension convert to uint8
    mask_image = Image.fromarray(np.array(mask.squeeze(0)).astype(np.uint8))
    # resize mask to size of the original image
    mask_image_resized = mask_image.resize((width,height))
    # convert resized mask to array
    mask_image_resized_array = np.array(mask_image_resized)
    
    # compute sums of predicted scores along the height of the image
    height_sums = np.apply_along_axis(np.sum,1,mask_image_resized_array)
    # get capital boundaries alongs the height axis
    height_caps = get_offsets(np.where(height_sums > 0)[0])

    capitals = [] # save here coordinates and image as array 
    
    for y_b,y_e in height_caps:
        # get chunk defined by y_axis coordinates
        chunk = mask_image_resized_array[y_b:y_e,:]
        # compute the sum of predictions along the x axis
        width_sum = np.apply_along_axis(np.sum,0,chunk)
        # get offsets
        width_caps = get_offsets(np.where(width_sum > 0)[0])
        if len(width_caps) > 1:
            print('This should not happen')
            print(path)
        x_b,x_e = width_caps[0]
        # save image along with coordinates
        capitals.append((y_b,y_e,x_b,x_e,image_array[y_b:y_e,x_b:x_e]))
        
    
    for (y_b,y_e,x_b,x_e,cap) in capitals:
        try:
            # we resize the image by quartering it
            # divide size by two for both x and y
            y_res = abs(int((y_b - y_e) / 2))
            x_res = abs(int((x_b - x_e) / 2))
            # turn array back into image
            cap_img = Image.fromarray(cap)
            # resize
            cap_img_res = cap_img.resize((x_res,y_res))
            # turn image back into array
            cap_img_res_arr = np.array(cap_img_res).copy()
            # the hook is the point that determins where
            # we will put the resized capital matrix
            hook_y = y_b
            hook_x = x_b + x_res
            # erase original capital with white space
            image_array[y_b:y_e-5,x_b:x_e-5] = 255
            # insert resized capital
            image_array[hook_y: hook_y + y_res,
                        hook_x: hook_x + x_res] = cap_img_res_arr
        except Exception as e:
            print("Other error:")
            print(e)
            print(path)
    
    return image_array

def save_capitals(image_paths: list, 
                model,
                ignore_files:list=[]):
    """function that accepts a list of paths to images,
    extracts, resizes capitals and saves the image. input is assumed to follow
    the way we currently store MDP data, processed images
    are stores in Image_processed_final folder
    
    Arguments:
        image_paths (list): list with paths to image files
        model (unet): trained unet model

    Returns:
        save the manipulated image
    """
    for p in tqdm(image_paths):
        if p in ignore_files: continue
        save_to = p.parent.parent / 'Image_processed_final'
        save_to.mkdir(exist_ok=True)
        
        path_img = save_to / f"{p.stem}.png"
        
        
        if path_img.is_file():
            continue

        try:
            image_manipulated = extract_and_adjust_capitals(p,model)
        except Exception as e:
            print(f"[WARNING] Error {e} for file {p.stem}")
            continue
        
        
        try:
            Image.fromarray(image_manipulated).save(path_img)
        except Exception as e:
            print(f"[WARNING] Error {e} for file {p.stem}")