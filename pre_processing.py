import tensorflow as tf
import tensorflow_datasets as tfds
import SimpleITK as sitk
import numpy as np
import scipy
from scipy import stats
from scipy import ndimage
from skimage.segmentation import flood, flood_fill
import psutil
import gc
import sys
import re

import os
import glob
import time
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['font.size'] = 5.0
from IPython.display import clear_output

np.random.seed(0)

the_scale = 0.5
lesions_compute = False
lesions_masks = False
out_folder = './pre_processed/'

if(len(sys.argv)>1):
    out_type = str(sys.argv[1])
    lim_inf = int(sys.argv[2])  # image array index to start
    lim_sup = int(sys.argv[3])#len(names) # image array index to end
else:
    lim_inf = 0 # image array index to start
    lim_sup = 148 # image array index to end
    out_type = "tmax_contralateral"


def load_vol_ITK(datafile):
    """
    load volume file
    formats: everything SimpleITK is able to read
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(datafile)
    itkImage = reader.Execute()

    return itkImage


def largest_dim_return(dimx, dimy, dimz, dimt, new_shape):

    if new_shape[0]>dimx:
        dimx = new_shape[0]
    if new_shape[1]>dimy:
        dimy = new_shape[1]
    if new_shape[2]>dimz:
        dimz = new_shape[2]
    if new_shape[3]>dimt:
        dimt = new_shape[3]

    return dimx, dimy, dimz, dimt


def verify_names(arrays_of_names, array_of_arrays):

    count = len(arrays_of_names[0])
    n_to_compare = len(arrays_of_names)
    for ar in range(n_to_compare):
        if len(arrays_of_names[ar]) != count:
            print("the number of datasets doesn't match.")
            break

    for n in range(count):
        nam = arrays_of_names[0][n]
        for ar in range(n_to_compare):
            if nam != arrays_of_names[ar][n]:
                print("the name " + nam + " doesn't match.")

    for n in range(count):
        shape = list(array_of_arrays[0][n].shape)
        for ar in range(n_to_compare):
            if shape[0] != array_of_arrays[ar][n].shape[0]:
                print("the shape of " + arrays_of_names[ar][n] + " doesn't match.")
                print(array_of_arrays[ar][n].shape)
                print(shape)

            if shape[1] != array_of_arrays[ar][n].shape[1]:
                print("the shape of " + arrays_of_names[ar][n] + " doesn't match.")
                print(array_of_arrays[ar][n].shape)
                print(shape)

            if shape[2] != array_of_arrays[ar][n].shape[2]:
                print("the shape of " + arrays_of_names[ar][n] + " doesn't match.")
                print(array_of_arrays[ar][n].shape)
                print(shape)

    print("Data integrity verified.")




def straighten_image(target, reference, type="4d"):
    theta = []
    cm = []
    for nn in range(reference.shape[0]):
        if (np.sum(reference[nn])>1):
            y, x = np.nonzero(reference[nn])
            x = x - np.nanmean(x)
            y = y - np.nanmean(y)
            coords = np.vstack([x, y])
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]
            x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
            x_v2, y_v2 = evecs[:, sort_indices[1]]
            theta.append(np.arctan((x_v1)/(y_v1)))
            cm.append(ndimage.measurements.center_of_mass(reference[nn]))
    theta = np.array(theta)
    theta = np.mean(theta)
    print(theta)

    cm = np.array(cm)
    cm = np.mean(cm, axis=0)
    print(cm)

    #rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    output = np.zeros(target.shape)
    if type=="4d":
        for sl in range(target.shape[0]):
            for tp in range(target.shape[3]):
                output[sl, :, :, tp] = ndimage.shift(target[sl, :, :, tp], list([int((target.shape[1]/2)-cm[0]),int((target.shape[2]/2)-cm[1])]), order=0)
                output[sl, :, :, tp] = ndimage.rotate(output[sl, :, :, tp], -np.rad2deg(theta), order=1, reshape=False)
    
    elif type=="3d":
        for sl in range(target.shape[0]):
            output[sl, :, :] = ndimage.shift(target[sl, :, :], list([int((target.shape[1]/2)-cm[0]),int((target.shape[2]/2)-cm[1])]), order=0)
            output[sl, :, :] = ndimage.rotate(output[sl, :, :], -np.rad2deg(theta), order=1, reshape=False)
    else:
        "Error: please specify a valid type for straighten_image() ex. 3d, 4d"

    return output


def correct_orientation(target, reference):
    is_left = 0
    is_right = 0
    for nn in range(reference.shape[0]):
        left = reference[nn, :, 0:int(reference.shape[2]/2)]
        right = reference[nn, :, int(reference.shape[2]/2):]

        left_len = 0
        for yy in reversed(range(left.shape[0])):
            if 1.0 in left[yy, :]:
                left_len = yy
        for yy in range(left.shape[0]):
            if 1.0 in left[yy, :]:
                left_len = left_len - yy

        right_len = 0
        for yy in reversed(range(right.shape[0])):
            if 1.0 in right[yy, :]:
                right_len = yy
        for yy in range(right.shape[0]):
            if 1.0 in right[yy, :]:
                right_len = right_len - yy

        if left_len <= right_len:
            is_left = is_left + 1
        else:
            is_right = is_right + 1

        if np.sum(left) >= np.sum(right):
            is_left = is_left + 1
        else:
            is_right = is_right + 1

    #print(is_right)
    #print(is_left)

    if is_right > is_left:
        return np.flip(target, 2)
    else:
        return target



def extract_names(folder, folder_name, file_name):
    names = glob.glob(folder, recursive=True)
    names = sorted(names)

    assert len(names) > 0, "Could not find any training data"
    print("Number of samples: ", len(names))

    names_extracted = []
    for name in names:
        result = re.search(folder_name+'(.*)'+file_name, name)
        names_extracted.append(result.group(1))

    return names, names_extracted


def print_ram():
    process_m = psutil.Process(os.getpid())
    print("RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))


# Define folder paths and files
scratch_path = "../Alejandro_Stroke"

folder = scratch_path + "/image_data/**/"+out_type+".npz"
lesion_folder = scratch_path + "/image_data/**/lesion.npz"
mask_folder = scratch_path + "/image_data/**/brain_mask_contralateral.npz"

folder_name = scratch_path + "/image_data/"

file_name = "/"+out_type+".npz"
lesion_name = "/lesion.npz"
mask_name = "/brain_mask_contralateral.npz"



# Extract names
names, names_extracted = extract_names(folder, folder_name, file_name)
lesion_names, lesion_names_extracted = extract_names(lesion_folder, folder_name, lesion_name)
mask_names, mask_names_extracted = extract_names(mask_folder, folder_name, mask_name)


# Load the arrays
names = names[lim_inf:lim_sup]
names_extracted = names_extracted[lim_inf:lim_sup]
ctp_samples = []
n = 0
dimx, dimy, dimz, dimt = [0, 0, 0, 0]
for name in names:
    ctp_samples.append(np.load(name)["arr_0"])
    print("{:.1f}".format(n*100/len(names))+"%")
    #dimx, dimy, dimz, dimt = largest_dim_return(dimx, dimy, dimz, dimt, ctp_samples[-1].shape)
    ctp_samples[-1] = ctp_samples[-1].astype("float32")
    #print(ctp_samples[-1].shape)
    n = n + 1

dimx, dimy, dimz, dimt = [32, 448, 320, 32]
print_ram()


lesion_names = lesion_names[lim_inf:lim_sup]
lesion_names_extracted = lesion_names_extracted[lim_inf:lim_sup]
ctp_lesions = []
n = 0
for name in lesion_names:
    ctp_lesions.append(np.load(name)["arr_0"])
    print("{:.1f}".format(n*100/len(lesion_names))+"%")
    #print(ctp_lesions[-1].shape)
    n = n + 1

print_ram()


mask_names = mask_names[lim_inf:lim_sup]
mask_names_extracted = mask_names_extracted[lim_inf:lim_sup]
ctp_masks = []
n = 0
for name in mask_names:
    ctp_masks.append(np.load(name)["arr_0"])
    print("{:.1f}".format(n*100/len(mask_names))+"%")
    n = n + 1

print_ram()


verify_names([names_extracted[lim_inf:lim_sup], lesion_names_extracted[lim_inf:lim_sup], mask_names_extracted[lim_inf:lim_sup]], [ctp_samples, ctp_lesions, ctp_masks])


# Harmonize the data.


if out_type == "interpolated_pwi":

        
    # Calculate padding for unique dimensions.
    ctp_samples_harmonized = np.zeros((len(ctp_samples), dimx, dimy, dimz, dimt), dtype="float32")
    if lesions_masks:
        ctp_lesions_harmonized = np.zeros((len(ctp_lesions), dimx, dimy, dimz), dtype="float32")
    
    ctp_masks_harmonized = np.zeros((len(ctp_masks), dimx, dimy, dimz), dtype="float32")

    for n in range(len(ctp_samples)):

        im = list(ctp_samples[n].shape)
        space_x, space_y, space_z, space_t = [0, 0, 0, 0]
        pad_x = math.floor((dimx - im[0])/2)
        pad_y = math.floor((dimy - im[1])/2)
        pad_z = math.floor((dimz - im[2])/2)
        pad_t = math.floor((dimt - im[3])/2)

        if pad_x<0:
            space_x = abs(pad_x)
            pad_x = 0
            im[0] = dimx
        if pad_y<0:
            space_y = abs(pad_y)
            pad_y = 0
            im[1] = dimy
        if pad_z<0:
            space_z = abs(pad_z)
            pad_z = 0
            im[2] = dimz
        if pad_t<0:
            space_t = abs(pad_t)
            pad_t = 0
            im[3] = dimt

        ctp_samples_harmonized[n, pad_x:pad_x+im[0], pad_y:pad_y+im[1], pad_z:pad_z+im[2], pad_t:pad_t+im[3]] = ctp_samples[n][space_x:im[0]+space_x, space_y:im[1]+space_y, space_z:im[2]+space_z, space_t:im[3]+space_t] 
        if lesions_masks:
            ctp_lesions_harmonized[n, pad_x:pad_x+im[0], pad_y:pad_y+im[1], pad_z:pad_z+im[2]] = ctp_lesions[n][space_x:im[0]+space_x, space_y:im[1]+space_y, space_z:im[2]+space_z]
        
        ctp_masks_harmonized[n, pad_x:pad_x+im[0], pad_y:pad_y+im[1], pad_z:pad_z+im[2]] = ctp_masks[n][space_x:im[0]+space_x, space_y:im[1]+space_y, space_z:im[2]+space_z]

    # Downscale
    if the_scale!=1.0:
        ctp_samples_harmonized = scipy.ndimage.interpolation.zoom(ctp_samples_harmonized, [1, 1, the_scale, the_scale, 1], order=1, mode='nearest')
        if lesions_masks:
             ctp_lesions_harmonized = scipy.ndimage.interpolation.zoom(ctp_lesions_harmonized, [1, 1, the_scale, the_scale], order=0, mode='nearest')
        
        ctp_masks_harmonized = scipy.ndimage.interpolation.zoom(ctp_masks_harmonized, [1, 1, the_scale, the_scale], order=0, mode='nearest')
           
else:


    # Calculate padding for unique dimensions.
    ctp_samples_harmonized = np.zeros((len(ctp_samples), dimx, dimy, dimz), dtype="float32")
    if lesions_masks:
        ctp_lesions_harmonized = np.zeros((len(ctp_lesions), dimx, dimy, dimz), dtype="float32")

    ctp_masks_harmonized = np.zeros((len(ctp_masks), dimx, dimy, dimz), dtype="float32")
        

    for n in range(len(ctp_samples)):

        im = list(ctp_samples[n].shape)
        space_x, space_y, space_z = [0, 0, 0]
        pad_x = math.floor((dimx - im[0])/2)
        pad_y = math.floor((dimy - im[1])/2)
        pad_z = math.floor((dimz - im[2])/2)

        if pad_x<0:
            space_x = abs(pad_x)
            pad_x = 0
            im[0] = dimx
        if pad_y<0:
            space_y = abs(pad_y)
            pad_y = 0
            im[1] = dimy
        if pad_z<0:
            space_z = abs(pad_z)
            pad_z = 0
            im[2] = dimz

        ctp_samples_harmonized[n, pad_x:pad_x+im[0], pad_y:pad_y+im[1], pad_z:pad_z+im[2]] = ctp_samples[n][space_x:im[0]+space_x, space_y:im[1]+space_y, space_z:im[2]+space_z] 
        if lesions_masks:
            ctp_lesions_harmonized[n, pad_x:pad_x+im[0], pad_y:pad_y+im[1], pad_z:pad_z+im[2]] = ctp_lesions[n][space_x:im[0]+space_x, space_y:im[1]+space_y, space_z:im[2]+space_z]
        
        ctp_masks_harmonized[n, pad_x:pad_x+im[0], pad_y:pad_y+im[1], pad_z:pad_z+im[2]] = ctp_masks[n][space_x:im[0]+space_x, space_y:im[1]+space_y, space_z:im[2]+space_z]

    # Downscale
    if the_scale!=1.0:
        ctp_samples_harmonized = scipy.ndimage.interpolation.zoom(ctp_samples_harmonized, [1, 1, the_scale, the_scale], order=1, mode='nearest')
        if lesions_masks:
            ctp_lesions_harmonized = scipy.ndimage.interpolation.zoom(ctp_lesions_harmonized, [1, 1, the_scale, the_scale], order=0, mode='nearest')

        ctp_masks_harmonized = scipy.ndimage.interpolation.zoom(ctp_masks_harmonized, [1, 1, the_scale, the_scale], order=0, mode='nearest')
            
print(ctp_samples_harmonized.shape)
if lesions_masks:
    print(ctp_lesions_harmonized.shape)

print(ctp_masks_harmonized.shape)

# Bounding box
"""
def get_edges(darr):
    start = 0
    end = 0
    for n in range(len(darr)):
        if darr[n]>0:
            start = n
            break

    for n in range(len(darr)):
        if darr[len(darr)-1-n]>0:
            end = len(darr)-n
            break

    return start, end


bby = np.sum(ctp_masks_harmonized, axis=(0, 1, 2))
bbx = np.sum(ctp_masks_harmonized, axis=(0, 1, 3))

print(bby.shape)

start_y, end_y = get_edges(bby)
print(start_y)
print(end_y)

start_x, end_x = get_edges(bbx)
print(start_x)
print(end_x)
"""

print_ram()

# Get statistics
if(out_type=="followup" or out_type=="perfusion_average" or out_type=="followup_contralateral"):
    ctp_samples_harmonized = np.where(ctp_samples_harmonized<0, 0, ctp_samples_harmonized)
    ctp_samples_harmonized = np.where(ctp_samples_harmonized>100, 0, ctp_samples_harmonized)
    
    ctp_samples_harmonized = ctp_samples_harmonized*ctp_masks_harmonized
    #min_dist = ctp_samples_harmonized.reshape(*ctp_samples_harmonized.shape[:-3], -1)
    #min_dist = min_dist.min(axis=1)
    #print(np.nanmean(min_dist))
    #print(min_dist.min())


if os.path.exists("norm_parameters_"+out_type+".csv"):
    na = np.genfromtxt("norm_parameters_"+out_type+".csv", delimiter=',')
    new_min = na[0]
    darange = na[1]
else:

    damean = np.nanmean(ctp_samples_harmonized[ctp_samples_harmonized>0])
    dastd = np.nanstd(ctp_samples_harmonized[ctp_samples_harmonized>0])
    damin = ctp_samples_harmonized.min()
    damax = ctp_samples_harmonized.max()

    if(out_type=="followup" or out_type=="perfusion_average" or out_type=="followup_contralateral"):

        if ((damean - (3*dastd)) < damin):
            new_min = damin
        else:
            new_min = damean - (3*dastd)

        if ((damean + (3*dastd)) > damax):
            new_max = damax
        else:
            new_max = damean + (3*dastd)

    else:

        if ((damean - dastd) < damin):
            new_min = damin
        else:
            new_min = damean - dastd

        if ((damean + dastd) > damax):
            new_max = damax
        else:
            new_max = damean + dastd

    darange = new_max - new_min
    na = np.asarray([new_min, darange])
    np.savetxt("norm_parameters_"+out_type+".csv", na, delimiter=",")
    print(damean)
    print(dastd)
    print(damin)
    print(damax)

    
print(new_min)
print(darange)



# Center and straighten
for n in range(ctp_samples_harmonized.shape[0]):
    if out_type=="interpolated_pwi":
        ctp_samples_harmonized[n] = straighten_image(ctp_samples_harmonized[n], ctp_masks_harmonized[n], type="4d")
    else:
        ctp_samples_harmonized[n] = straighten_image(ctp_samples_harmonized[n], ctp_masks_harmonized[n], type="3d")
    
    if lesions_masks:
        ctp_lesions_harmonized[n] = straighten_image(ctp_lesions_harmonized[n], ctp_masks_harmonized[n], type="3d")
    
    ctp_masks_harmonized[n] = straighten_image(ctp_masks_harmonized[n], ctp_masks_harmonized[n], type="3d")
    
    ctp_samples_harmonized[n] = correct_orientation(ctp_samples_harmonized[n], ctp_masks_harmonized[n])
    
    if lesions_masks:
        ctp_lesions_harmonized[n] = correct_orientation(ctp_lesions_harmonized[n], ctp_masks_harmonized[n])
    
    ctp_masks_harmonized[n] = correct_orientation(ctp_masks_harmonized[n], ctp_masks_harmonized[n])

    print_ram()
    print(str(n)+" straightening and centering completed")

#ctp_samples_harmonized = np.where(ctp_samples_harmonized >= new_max, new_max, ctp_samples_harmonized)
#ctp_samples_harmonized = np.where(ctp_samples_harmonized <= new_min, new_min, ctp_samples_harmonized)

# Normalize and clip


ctp_samples = np.array((((ctp_samples_harmonized-new_min)/darange)*2)-1).astype("float32")
if(out_type=="followup" or out_type=="perfusion_average" or out_type=="interpolated_pwi" or out_type=="followup_contralateral"):
    ctp_samples = np.where(ctp_samples<-1.0, -1.0, ctp_samples)
    ctp_samples = np.where(ctp_samples>1.0, 1.0, ctp_samples)
print(ctp_samples.shape)
print(ctp_samples.min())
print(ctp_samples.max())


if lesions_masks:
    ctp_lesions = np.array(ctp_lesions_harmonized).astype("float32")
    print(ctp_lesions.shape)

ctp_masks = np.array(ctp_masks_harmonized).astype("float32")
print(ctp_masks.shape)

print_ram()


for n in range(len(names)):
    with open(out_folder+"npy/"+out_type+"/"+names_extracted[n]+".npy", 'wb') as f:
        np.save(f, ctp_samples[n])

    img = sitk.GetImageFromArray(ctp_samples[n])
    sitk.WriteImage(img, out_folder+"nii/"+out_type+"/"+names_extracted[n]+".nii.gz")

if lesions_compute:
    for n in range(len(names)):
        with open(out_folder+"npy/lesions/"+names_extracted[n]+".npy", 'wb') as f:
            np.save(f, ctp_lesions[n])

        img = sitk.GetImageFromArray(ctp_lesions[n])
        sitk.WriteImage(img, out_folder+"nii/lesions/"+names_extracted[n]+".nii.gz")

    for n in range(len(names)):
        with open(out_folder+"npy/masks/"+names_extracted[n]+".npy", 'wb') as f:
            np.save(f, ctp_masks[n])

        img = sitk.GetImageFromArray(ctp_masks[n])
        sitk.WriteImage(img, out_folder+"nii/masks/"+names_extracted[n]+".nii.gz")


# Plot images
upper_s = 16
lower_s = 8
upper_t = 15
lower_t = 6
slices = upper_s - lower_s + 1
timepoints = upper_t - lower_t + 1

if out_type == "interpolated_pwi":
    for n in range(len(names)):
        image = ctp_samples[n]
        plt.figure(dpi=300)#figsize=(32, slices*1.2), dpi=300)
        for daslice in range(lower_s, upper_s):#slices): 
            for timepoint in range(lower_t, upper_t):#32):
                level = (daslice-lower_s) * timepoints#daslice * 32
                if daslice == lower_s:#0:
                    plt.subplot(slices, timepoints, (level+timepoint-lower_t+1), title=str(timepoint), aspect='auto')
                else:
                    plt.subplot(slices, timepoints, (level+timepoint-lower_t+1), aspect='auto')
                #mid = int((int(image.shape[0])/2) - 1)
                plt.imshow(image[daslice, :, :, timepoint], cmap='gray', vmin=-1, vmax=1.0)
                plt.axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle(names_extracted[n], color='white')
        plt.savefig(out_folder+"jpg/"+out_type+"/"+names_extracted[n]+".jpg", bbox_inches='tight', facecolor='black')
        plt.close()
        process_m = psutil.Process(os.getpid())
        print(str(n)+" - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

else:
    for n in range(len(names)):
        image = ctp_samples[n]
        plt.figure(dpi=300)#figsize=(32, slices*1.2), dpi=300)

        for daslice in range(image.shape[0]):#slices): 
            plt.subplot(6, 6, daslice+1)
            plt.imshow(image[daslice, :, :], cmap='gray', vmin=-1, vmax=1.0)
            plt.axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle(names_extracted[n], color='white')
        plt.savefig(out_folder+"jpg/"+out_type+"/"+names_extracted[n]+".jpg", bbox_inches='tight', facecolor='black')
        plt.close()
        process_m = psutil.Process(os.getpid())
        print(str(n)+" - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

if lesions_compute:
    for n in range(len(names)):
        image = ctp_lesions[n]
        plt.figure(dpi=300)#figsize=(32, slices*1.2), dpi=300)
        daslice = 16 
        #mid = int((int(image.shape[0])/2) - 1)
        plt.imshow(image[daslice, :, :], cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.suptitle(names_extracted[n], color='white')
        plt.savefig(out_folder+"jpg/lesions/"+names_extracted[n]+".jpg", bbox_inches='tight')
        plt.close()
        process_m = psutil.Process(os.getpid())
        print(str(n)+" - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

    for n in range(len(names)):
        image = ctp_masks[n]
        plt.figure(dpi=300)#figsize=(32, slices*1.2), dpi=300)
        daslice = 16 
        #mid = int((int(image.shape[0])/2) - 1)
        plt.imshow(image[daslice, :, :], cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.suptitle(names_extracted[n], color='white')
        plt.savefig(out_folder+"jpg/masks/"+names_extracted[n]+".jpg", bbox_inches='tight')
        plt.close()
        process_m = psutil.Process(os.getpid())
        print(str(n)+" - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))


"""
for n in range(len(names)):
    image = ctp_samples[n]
    lesion = ctp_lesions[n]
    plt.figure(dpi=300)#figsize=(32, slices*1.2), dpi=300)
    for daslice in range(lower_s, upper_s):#slices): 
        for timepoint in range(lower_t, upper_t):#32):
            level = (daslice-lower_s) * timepoints#daslice * 32
            if daslice == lower_s:#0:
                plt.subplot(slices, timepoints, (level+timepoint-lower_t+1), title=str(timepoint), aspect='auto')
            else:
                plt.subplot(slices, timepoints, (level+timepoint-lower_t+1), aspect='auto')
            #mid = int((int(image.shape[0])/2) - 1)
            base = (image[daslice, :, :, timepoint])
            base = np.where(base >= 3.0, 3.0, base)
            base = np.interp(base, (base.min(), 3.0), (0, 1))
            overlay = np.array([base+(lesion[daslice, :, :]*0.15), base, base])
            overlay = np.where(overlay >= 1.0, 1.0, overlay)
            overlay = np.moveaxis(overlay, 0, -1)

            plt.imshow(overlay)
            plt.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(names_extracted[n], color='white')
    plt.savefig("inspect_images/"+"{:03d}".format(n)+"_"+names_extracted[n]+"_lesion.jpg", bbox_inches='tight', facecolor='black')
    plt.close()
    process_m = psutil.Process(os.getpid())
    print(str(n)+" - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

    """