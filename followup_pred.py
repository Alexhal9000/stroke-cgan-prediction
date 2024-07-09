import tensorflow as tf
import tensorflow_datasets as tfds
import SimpleITK as sitk
import pix2pix as pix2pix
import numpy as np
import scipy
from scipy import stats
from scipy import ndimage
import skimage
from skimage.segmentation import flood, flood_fill
import elastic_augmentation as ea
import random
import glob
import re
import psutil
import metrics
import seglosses
import sys
import pprint
import os
import time
import pickle
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

# load scans NCCT and FLAIR

LAMBDA = 1.0 # 1
LAMBDA_M = 0.3
#LAMBDA_B = 0.0 #0.5
LAMBDA_C = .01
#LAMBDA_D = 1

EPOCHS = 600

learning_rate = 2e-4 # last: 2e-5, 7e-5, 2e-4

scale_factor = 1
target_scale = 192
size_offset = 0

use_pmaps = False
use_avg = True
use_avg_only = False
pred_mask = False

treatment = "IVT"

if use_pmaps:

    if treatment == "IVT":
        folder_prepend = "./eval_1/auto_final_pmaps_avg/"
    else:
        folder_prepend = "./eval_1/auto_final_IA_pmaps/"

else:

    if treatment == "IVT":
        folder_prepend = "./eval_1/auto_final/"
    else:
        folder_prepend = "./eval_1/auto_final_IA/"

np.random.seed(10)
random.seed(10)

if treatment == "IVT":

    with open('./partition_10FCV_IV_2D.pickle', 'rb') as output:
        partition = pickle.load(output)

else:

    with open('./partition_10FCV_IA.pickle', 'rb') as output:
        partition = pickle.load(output)


if(len(sys.argv)>1):
    dafold = str(sys.argv[1])
else:
    dafold = 0

print(dafold)
folds = [*range(10)]
for j in range(2*int(dafold)):
    folds.append(folds.pop(0))
print(folds)

#pprint.pprint(partition)

nams_tr = []
for fold in range(8):
    for s in partition['testing'][folds[fold]]:
        if treatment == "IVT":
            ss = s.split("_", 1)[1]
            if ss not in nams_tr:
                nams_tr.append(ss)
        else:
            ss = s
            if ss not in nams_tr:
                nams_tr.append(ss)

print(nams_tr)

nams_test = []
for fold in range(8, 10):
    for s in partition['testing'][folds[fold]]:
        if treatment == "IVT":
            ss = s.split("_", 1)[1]
            if ss not in nams_test:
                nams_test.append(ss)
        else:
            ss = s
            if ss not in nams_test:
                nams_test.append(ss)
                
print(nams_test)



def load_vol_ITK(datafile):
    """
    load volume file
    formats: everything SimpleITK is able to read
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(datafile)
    itkImage = reader.Execute()

    return itkImage


def print_ram():
    process_m = psutil.Process(os.getpid())
    print("RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))


def extract_names(folder, folder_name, file_name):
    names = glob.glob(folder, recursive=False)
    names = sorted(names)

    assert len(names) > 0, "Could not find any training data"
    print("Number of samples: ", len(names))

    names_extracted = []
    for name in names:
        result = re.search(folder_name+'(.*)'+file_name, name)
        names_extracted.append(result.group(1))

    return names, names_extracted


def getimg(name):
    aa = np.load(name)[np.newaxis, ..., np.newaxis].astype("float32")
    return aa


def getparams(name, num):
    return np.concatenate((getimg(name[0][num]), getimg(name[1][num]), getimg(name[2][num]), getimg(name[3][num])), axis=4)


def NormalizeData(data):
    return ((data - np.min(data)) / (np.max(data) - np.min(data))*2) - 1.0

def getseverity(atmax, alesion, nam):

    atmax = NormalizeData(atmax)

    amin = 0
    ath = 0
    for th in range(2000):
        thr = th*(2/2000) 
        im1 = np.where((atmax+1)>thr, 1.0, 0.0)
        intersection = (im1*alesion)
        new = 2 * intersection.sum() / (im1.sum() + alesion.sum())
        if new > amin:
            amin = new
            ath = thr

    bmin = 0
    bth = 0
    aim1 = np.where((atmax+1)>ath, 1.0, 0.0)
    for th in range(2000):
        thr = th*(2/2000) 
        im1 = np.where((atmax+1)<thr, 1.0, 0.0)
        im1 = im1*aim1
        intersection = (im1*alesion)
        new = 2 * intersection.sum() / (im1.sum() + alesion.sum())
        if new > bmin:
            bmin = new
            bth = thr


    image_out = np.where((atmax+1)>ath, atmax, ath-1)
    image_out = np.where((image_out+1)<bth, image_out, bth-1)

    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.imshow(atmax[16, :, :])
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("threshold")
    plt.imshow(image_out[16, :, :])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("gt - "+str(round(bmin, 2)))
    plt.imshow(alesion[16, :, :])
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(folder_prepend + "thresholds_"+nam+".jpg", dpi=300)
    plt.close()


    return [amin, bmin, ath, bth]


def process_params(data):
    out = NormalizeData(((NormalizeData(np.where((data[0, :, :, :, 0]+
        1)<0.10, 0.10-(data[0, :, :, :, 0]+1), 0.0))+1)*
        (data[0, :, :, :, 3]+1))+(data[0, :, :, :, 2]+1))
    return out[np.newaxis, ..., np.newaxis]


names_pwi, names_extracted_pwi = extract_names("./pre_processed/npy/interpolated_pwi/*","pwi/",".npy")
names_followup, names_extracted_followup = extract_names("./pre_processed/npy/followup/*","followup/",".npy")
names_lesions, names_extracted_lesions = extract_names("./pre_processed/npy/lesions/*","lesions/",".npy")
names_masks, names_extracted_masks = extract_names("./pre_processed/npy/masks/*","masks/",".npy")
names_avg, names_extracted_avg = extract_names("./pre_processed/npy/perfusion_average/*","perfusion_average/",".npy")

names_cbf, names_extracted_cbf = extract_names("./pre_processed/npy/cbf/*","cbf/",".npy")
names_cbv, names_extracted_cbv = extract_names("./pre_processed/npy/cbv/*","cbv/",".npy")
names_mtt, names_extracted_mtt = extract_names("./pre_processed/npy/mtt/*","mtt/",".npy")
names_tmax, names_extracted_tmax = extract_names("./pre_processed/npy/tmax/*","tmax/",".npy")


src_len = len(names_followup)
src_order = np.arange(src_len)

np.random.shuffle(src_order)
print(src_order)

BUFFER_SIZE = 10
BATCH_SIZE = 1
OUTPUT_CHANNELS = 1
INPUT_CHANNELS = 4
EXTRA_CHANNELS = 1


params = [names_cbf, names_cbv, names_mtt, names_tmax]
source_images_params = [None]*4
source_test_params = [None]*4
source_val_params = [None]*4

param_type = ["cbf", "cbv", "mtt", "tmax"]


for param in range(len(params)):
    # -------- Load Pmaps ------

    params[param] = np.array(params[param])
    source_images_params_all = params[param][src_order]


    # do training validation testing split
    """
    source_val_params[param] = source_images_params_all[(math.floor(src_len*0.7)):(math.floor(src_len*0.85))]

    source_test_params[param] = source_images_params_all[(math.floor(src_len*0.85)):src_len]

    source_images_params[param] = source_images_params_all[0:(math.floor(src_len*0.7))]
    """

    source_test_params[param] = np.array(["./pre_processed/npy/"+param_type[param]+"/" + s + ".npy" for s in nams_test])
    source_images_params[param] = np.array(["./pre_processed/npy/"+param_type[param]+"/" + s + ".npy" for s in nams_tr])


    print("Source shape params "+str(param)+":")
    print(source_images_params[param].shape)

    #print("Validation source shape params "+str(param)+":")
    #print(source_val_params[param].shape)

    print("Test source shape params "+str(param)+":")
    print(source_test_params[param].shape)



sample_src_params = getparams(source_images_params, 9)

val_sample_src_params = getparams(source_test_params, 3)

print("Source sample shape params:")
print(sample_src_params.shape)


# ------------------------------------
# -------- Load pwi ------

names_pwi = np.array(names_pwi)
source_images_pwi = names_pwi[src_order]


# do training validation testing split
"""
source_val_pwi = source_images_pwi[(math.floor(src_len*0.7)):(math.floor(src_len*0.85))]

source_test_pwi = source_images_pwi[(math.floor(src_len*0.85)):src_len]

source_images_pwi = source_images_pwi[0:(math.floor(src_len*0.7))]
"""

source_test_pwi = np.array(["./pre_processed/npy/interpolated_pwi/" + s + ".npy" for s in nams_test])
source_images_pwi = np.array(["./pre_processed/npy/interpolated_pwi/" + s + ".npy" for s in nams_tr])



print("Source shape pwi:")
print(source_images_pwi.shape)

#print("Validation source shape pwi:")
#print(source_val_pwi.shape)

print("Test source shape pwi:")
print(source_test_pwi.shape)


sample_src_pwi = getimg(source_images_pwi[9])

val_sample_src_pwi = getimg(source_test_pwi[3])

print("Source sample shape pwi:")
print(sample_src_pwi.shape)


# ------------------------------------
# -------- Load Followup ------

names_followup = np.array(names_followup)
source_images_followup = names_followup[src_order]


# do training validation testing split
"""
source_val_followup = source_images_followup[(math.floor(src_len*0.7)):(math.floor(src_len*0.85))]

source_test_followup = source_images_followup[(math.floor(src_len*0.85)):src_len]

source_images_followup = source_images_followup[0:(math.floor(src_len*0.7))]
"""

source_test_followup = np.array(["./pre_processed/npy/followup/" + s + ".npy" for s in nams_test])
source_images_followup = np.array(["./pre_processed/npy/followup/" + s + ".npy" for s in nams_tr])


print("Source shape followup:")
print(source_images_followup.shape)

#print("Validation source shape followup:")
#print(source_val_followup.shape)

print("Test source shape followup:")
print(source_test_followup.shape)


sample_src_followup = getimg(source_images_followup[9])

val_sample_src_followup = getimg(source_test_followup[3])

print("Source sample shape followup:")
print(sample_src_followup.shape)

# ------------------------------------
# -------- Load Lesions ------

names_lesions = np.array(names_lesions)
source_images_lesions = names_lesions[src_order]


# do training validation testing split
"""
source_val_lesions = source_images_lesions[(math.floor(src_len*0.7)):(math.floor(src_len*0.85))]

source_test_lesions = source_images_lesions[(math.floor(src_len*0.85)):src_len]

source_images_lesions = source_images_lesions[0:(math.floor(src_len*0.7))]
"""

source_test_lesions = np.array(["./pre_processed/npy/lesions/" + s + ".npy" for s in nams_test])
source_images_lesions = np.array(["./pre_processed/npy/lesions/" + s + ".npy" for s in nams_tr])


print("Source shape lesions:")
print(source_images_lesions.shape)

#print("Validation source shape lesions:")
#print(source_val_lesions.shape)

print("Test source shape lesions:")
print(source_test_lesions.shape)


sample_src_lesions = getimg(source_images_lesions[9])

val_sample_src_lesions = getimg(source_test_lesions[3])

print("Source sample shape lesions:")
print(sample_src_lesions.shape)

# ------------------------------------
# -------- Load Masks ------

names_masks = np.array(names_masks)
source_images_masks = names_masks[src_order]


# do training validation testing split
"""
source_val_masks = source_images_masks[(math.floor(src_len*0.7)):(math.floor(src_len*0.85))]

source_test_masks = source_images_masks[(math.floor(src_len*0.85)):src_len]

source_images_masks = source_images_masks[0:(math.floor(src_len*0.7))]
"""
source_test_masks = np.array(["./pre_processed/npy/masks/" + s + ".npy" for s in nams_test])
source_images_masks = np.array(["./pre_processed/npy/masks/" + s + ".npy" for s in nams_tr])


print("Source shape masks:")
print(source_images_masks.shape)

#print("Validation source shape masks:")
#print(source_val_masks.shape)

print("Test source shape masks:")
print(source_test_masks.shape)


sample_src_masks = getimg(source_images_masks[9])

val_sample_src_masks = getimg(source_test_masks[3])

print("Source sample shape masks:")
print(sample_src_masks.shape)

# ------------------------------------
# -------- Load Avg ------

names_avg = np.array(names_avg)
source_images_avg = names_avg[src_order]


# do training validation testing split
"""
source_val_avg = source_images_avg[(math.floor(src_len*0.7)):(math.floor(src_len*0.85))]

source_test_avg = source_images_avg[(math.floor(src_len*0.85)):src_len]

source_images_avg = source_images_avg[0:(math.floor(src_len*0.7))]
"""

source_test_avg = np.array(["./pre_processed/npy/perfusion_average/" + s + ".npy" for s in nams_test])
source_images_avg = np.array(["./pre_processed/npy/perfusion_average/" + s + ".npy" for s in nams_tr])


print("Source shape avg:")
print(source_images_avg.shape)

#print("Validation source shape avg:")
#print(source_val_avg.shape)

print("Test source shape avg:")
print(source_test_avg.shape)


sample_src_avg = getimg(source_images_avg[9])

val_sample_src_avg = getimg(source_test_avg[3])

print("Source sample shape avg:")
print(sample_src_avg.shape)

# ------------------------------------

#seg_unet = tf.keras.models.load_model(folder_prepend+'seg_unet') 
#print(seg_unet.summary())

generator = pix2pix.unet_generator_3d_resnet(OUTPUT_CHANNELS, INPUT_CHANNELS + EXTRA_CHANNELS, norm_type='instancenorm')
print(generator.summary())

discriminator = pix2pix.discriminator_3d(INPUT_CHANNELS + EXTRA_CHANNELS + OUTPUT_CHANNELS, norm_type='instancenorm')
#print(discriminator.summary())

encoder = tf.keras.models.load_model('./eval_1/autoencoder_4/'+str(dafold)+'/encoder')#folder_prepend + '/encoder')
print(encoder.summary())


print(sample_src_pwi[:, 16, :, :, :, :].shape)
to_tar = encoder(sample_src_pwi[:, 16, :, :, :, :])
print(to_tar.shape)

plt.subplot(3, 4, 1)
plt.title("2D+t")
plt.imshow(sample_src_pwi[0, 16, :, :, 16, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 2)
plt.title("to 2D")
plt.imshow(np.sum(to_tar[0, :, :, 0, :], axis=2))
plt.axis('off')

plt.subplot(3, 4, 3)
plt.title("followup")
plt.imshow(sample_src_followup[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 4)
plt.title("lesions")
plt.imshow(sample_src_lesions[0, 16, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 5)
plt.title("masks")
plt.imshow(sample_src_masks[0, 16, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 6)
plt.title("avg")
plt.imshow(sample_src_avg[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 7)
plt.title("cbf")
plt.imshow(sample_src_params[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 8)
plt.title("cbv")
plt.imshow(sample_src_params[0, 16, :, :, 1], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 9)
plt.title("mtt")
plt.imshow(sample_src_params[0, 16, :, :, 2], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')

plt.subplot(3, 4, 10)
plt.title("tmax")
plt.imshow(sample_src_params[0, 16, :, :, 3], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')

#plt.show()
plt.tight_layout()
plt.savefig(folder_prepend + str(dafold)+"/"+ "loaded_modals.png")
plt.close()




print(names_extracted_tmax[9])
#sample_src_severity = getseverity(sample_src_params[0, :, :, :, 2], sample_src_lesions[0, :, :, :, 0])
#print("dice a "+str(sample_src_severity[0]) +" dice b "+str(sample_src_severity[1]) + " - th_a:" + str(sample_src_severity[2])+ " th_b:" + str(sample_src_severity[3]))

#sample_src_severity = getseverity(sample_src_params[0, :, :, :, 3]-sample_src_params[0, :, :, :, 0], sample_src_lesions[0, :, :, :, 0])
#print("dice a "+str(sample_src_severity[0]) +" dice b "+str(sample_src_severity[1]) + " - th_a:" + str(sample_src_severity[2])+ " th_b:" + str(sample_src_severity[3]))

"""
for n in range(11):
    sample_src_params = getparams(source_images_params, n)
    sample_src_lesions = getimg(source_images_lesions[n])
    sample_src_severity = getseverity(((NormalizeData(np.where((sample_src_params[0, :, :, :, 0]+
        1)<0.10, 0.10-(sample_src_params[0, :, :, :, 0]+1), 0.0))+1)*
        (sample_src_params[0, :, :, :, 3]+1))+(sample_src_params[0, :, :, :, 2]+1), sample_src_lesions[0, :, :, :, 0], str(n))
    #print("dice a "+str(sample_src_severity[0]) +" dice b "+str(sample_src_severity[1]) + " - th_a:" + str(sample_src_severity[2])+ " th_b:" + str(sample_src_severity[3]))
"""


#mydice = metrics.Dice(nb_labels=2, dice_type='soft')
#dice_loss = mydice.loss

#asym_unified_focal_loss = seglosses.asym_unified_focal_loss()


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def l1_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return loss1


def NGF(A, B):

    # sigma 5
    sigma = 1.0#2.0     # width of kernel
    x = np.arange(-3,4,1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel_g = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel_g = kernel_g / np.sum(kernel_g)

    kernel_g[0:3, :, :] = 0
    kernel_g[4:7, :, :] = 0

    kernel_gaussian = tf.constant(kernel_g, dtype=tf.float32)
    kernel_gaussian = kernel_gaussian[..., np.newaxis, np.newaxis]

    #print(kernel_gaussian)

    #print(kernel_gaussian.shape)

    A = tf.nn.conv3d(A, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    B = tf.nn.conv3d(B, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')


    dx = (A[:, :, 1:, 1:, :] - A[:, :, :-1, 1:, :]) 
    dy = (A[:, :, 1:, 1:, :] - A[:, :, 1:, :-1, :])

    #print(dx.shape)

    norm = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + .02**2)

    ngf_dx_A = tf.pad(dx / norm, tf.constant([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]]))
    ngf_dy_A = tf.pad(dy / norm, tf.constant([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]]))


    """
    print(ngf_dx.shape)

    plt.subplot(2, 2, 1)
    plt.imshow(ngf_dx[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(ngf_dy[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')
    """

    dx = (B[:, :, 1:, 1:, :] - B[:, :, :-1, 1:, :]) 
    dy = (B[:, :, 1:, 1:, :] - B[:, :, 1:, :-1, :])

    #print(dx.shape)

    norm = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + .02**2) # 1e-2

    ngf_dx_B = tf.pad(dx / norm, tf.constant([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]]))
    ngf_dy_B = tf.pad(dy / norm, tf.constant([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]]))

    
    """
    print(ngf_dx_B.shape)

    plt.subplot(2, 2, 3)
    plt.imshow(ngf_dx_B[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(ngf_dy_B[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.savefig(folder_prepend + "gradient_ngf.jpg", dpi=300)
    plt.close()
    """

    

    return (1/2)*(NCC(ngf_dx_A, ngf_dx_B)+NCC(ngf_dy_A, ngf_dy_B))




def GC(A, B):

    # sigma 5
    sigma = 1.0     # width of kernel
    x = np.arange(-3,4,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel_g = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel_g = kernel_g / np.sum(kernel_g)

    kernel_gaussian = tf.constant(kernel_g, dtype=tf.float32)
    kernel_gaussian = kernel_gaussian[..., np.newaxis, np.newaxis]

    grad_op_A = A
    grad_op_B = B

    grad_op_A = tf.nn.conv3d(grad_op_A, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_B = tf.nn.conv3d(grad_op_B, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')

    #grad_op_A = tf.nn.conv3d(grad_op_A, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    #grad_op_B = tf.nn.conv3d(grad_op_B, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')

    kernel_in_x = np.array([ [
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        [ [ [-4] ],[ [0] ],[ [4] ] ],
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ]])
    kernel_x = tf.constant(kernel_in_x, dtype=tf.float32)
    #kernel_x = kernel_x[..., np.newaxis, np.newaxis]

    kernel_in_y = np.array([ [
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [1] ],[ [4] ],[ [1] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [-1] ],[ [-4] ],[ [-1] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ]])
    kernel_y = tf.constant(kernel_in_y, dtype=tf.float32)
    #kernel_y = kernel_y[..., np.newaxis, np.newaxis]

    kernel_in_z = np.array([ [
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ]])
    kernel_z = tf.constant(kernel_in_z, dtype=tf.float32)
    #kernel_z = kernel_z[..., np.newaxis, np.newaxis]

    grad_op_x_A = tf.nn.conv3d(grad_op_A, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_x_B = tf.nn.conv3d(grad_op_B, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_A = tf.nn.conv3d(grad_op_A, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_B = tf.nn.conv3d(grad_op_B, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_A = tf.nn.conv3d(grad_op_A, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_B = tf.nn.conv3d(grad_op_B, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')


    grad_op_x_A = grad_op_x_A
    grad_op_x_B = grad_op_x_B
    grad_op_y_A = grad_op_y_A
    grad_op_y_B = grad_op_y_B
    grad_op_z_A = grad_op_z_A
    grad_op_z_B = grad_op_z_B

    return (1/3)*(NCC(grad_op_x_A, grad_op_x_B)+NCC(grad_op_y_A, grad_op_y_B)+NCC(grad_op_z_A, grad_op_z_B))


def GC_show(A, B):

    # sigma 5
    sigma = 1.0     # width of kernel
    x = np.arange(-3,4,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel_g = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel_g = kernel_g / np.sum(kernel_g)

    kernel_gaussian = tf.constant(kernel_g, dtype=tf.float32)
    kernel_gaussian = kernel_gaussian[..., np.newaxis, np.newaxis]

    grad_op_A = A
    grad_op_B = B

    grad_op_A = tf.nn.conv3d(grad_op_A, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_B = tf.nn.conv3d(grad_op_B, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')

    #grad_op_A = tf.nn.conv3d(grad_op_A, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    #grad_op_B = tf.nn.conv3d(grad_op_B, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')

    kernel_in_x = np.array([ [
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        [ [ [-4] ],[ [0] ],[ [4] ] ],
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ]])
    kernel_x = tf.constant(kernel_in_x, dtype=tf.float32)
    #kernel_x = kernel_x[..., np.newaxis, np.newaxis]

    kernel_in_y = np.array([ [
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [1] ],[ [4] ],[ [1] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [-1] ],[ [-4] ],[ [-1] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ]])
    kernel_y = tf.constant(kernel_in_y, dtype=tf.float32)
    #kernel_y = kernel_y[..., np.newaxis, np.newaxis]

    kernel_in_z = np.array([ [
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ]])
    kernel_z = tf.constant(kernel_in_z, dtype=tf.float32)
    #kernel_z = kernel_z[..., np.newaxis, np.newaxis]

    grad_op_x_A = tf.nn.conv3d(grad_op_A, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_x_B = tf.nn.conv3d(grad_op_B, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_A = tf.nn.conv3d(grad_op_A, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_B = tf.nn.conv3d(grad_op_B, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_A = tf.nn.conv3d(grad_op_A, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_B = tf.nn.conv3d(grad_op_B, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')


    grad_op_x_A = grad_op_x_A
    grad_op_x_B = grad_op_x_B
    grad_op_y_A = grad_op_y_A
    grad_op_y_B = grad_op_y_B
    grad_op_z_A = grad_op_z_A
    grad_op_z_B = grad_op_z_B

    plt.subplot(2, 3, 1)
    plt.imshow(grad_op_x_A[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(grad_op_y_A[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(A[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(grad_op_x_B[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(grad_op_y_B[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(B[0, 16, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.savefig(folder_prepend + "gradient.jpg", dpi=300)
    plt.close()




def NCC(A, B):
    A_norm = A-tf.reduce_mean(A)
    B_norm = B-tf.reduce_mean(B)
    A_redux = tf.math.reduce_sum(A_norm*A_norm)
    B_redux = tf.math.reduce_sum(B_norm*B_norm)
    AB_root = tf.math.sqrt(A_redux*B_redux)
    
    return tf.math.reduce_sum(A_norm*B_norm)/(AB_root+0.0000000000000001)#((A_root*B_root)+0.00000000001)



#sample_src_lesions = tf.keras.utils.to_categorical(sample_src_lesions[:, :, :, :, 0], num_classes=2, dtype='float32')
#val_sample_src_lesions = tf.keras.utils.to_categorical(val_sample_src_lesions[:, :, :, :, 0], num_classes=2, dtype='float32')
#sample_src_masks = tf.keras.utils.to_categorical(sample_src_masks[:, :, :, :, 0], num_classes=2, dtype='float32')

print(sample_src_lesions.shape)

#print(dice_loss(sample_src_lesions, sample_src_lesions))
#print(dice_loss(sample_src_lesions, sample_src_masks))

#print(asym_unified_focal_loss(sample_src_lesions, sample_src_lesions))
#print(asym_unified_focal_loss(sample_src_lesions, sample_src_masks))

"""
plt.title("loaded seg unet")
#print(sample_src_followup.shape)
#print(seg_unet(sample_src_followup).shape)
plt.imshow(seg_unet(sample_src_followup)[0, 16, :, :, 1], cmap='gray', vmin=0.0, vmax=1.0)
plt.axis('off')
plt.tight_layout()
plt.savefig(folder_prepend + "initial_state_seg.png")
plt.close()
"""

def encode_vol(pwi):
    out = np.empty((32, 224, 160, 4))
    for a in range(32):
        #print(encoder(sample_src_pwi[:, a, :, :, :, :]).shape)
        out[a, :, :, :] = encoder(pwi[:, a, :, :, :, :])[0, :, :, :, 0]
    #print("out shape: "+out.shape)
    out = out[np.newaxis, ...]
    return out.astype("float32")


plt.title("initial test followup pred")
if use_avg_only:
    sample_src_con = sample_src_avg
    val_sample_src_con = val_sample_src_avg
else:
    if use_pmaps:
        if use_avg:
            sample_src_con = np.concatenate((sample_src_params, sample_src_avg), axis=4, dtype="float32")
            val_sample_src_con = np.concatenate((val_sample_src_params, val_sample_src_avg), axis=4, dtype="float32")
        else:
            sample_src_con = sample_src_params
            val_sample_src_con = val_sample_src_params
    else:
        if use_avg:
            sample_src_con = np.concatenate((encode_vol(sample_src_pwi), sample_src_avg), axis=4, dtype="float32")
            val_sample_src_con = np.concatenate((encode_vol(val_sample_src_pwi), val_sample_src_avg), axis=4, dtype="float32")
        else:
            sample_src_con = encode_vol(sample_src_pwi)
            val_sample_src_con = encode_vol(val_sample_src_pwi)

plt.imshow(generator(sample_src_con)[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.axis('off')
plt.tight_layout()
plt.savefig(folder_prepend +"/"+str(dafold)+"/" + "initial_state_followup.png")
plt.close()


generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999)


def generate_images(model, test_input, ground_truth, label, ground_truth_mask=0):

    to_tar = model(test_input)

    if pred_mask:

        if not use_avg_only:
            plt.subplot(2, 3, 1)
            plt.title("compressed pwi")
            imm = np.moveaxis(np.array([test_input[0, 16, :, :, 0], test_input[0, 16, :, :, 1], test_input[0, 16, :, :, 3]]), 0, -1)
            plt.imshow(imm)#test_input[0, 16, :, :, 0], vmin=-1.0, vmax=1.0)#np.sum(test_input[0, 16, :, :, :], axis=2))
            plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title("avg")
        plt.imshow(test_input[0, 16, :, :, -1], cmap='gray', vmin=-1.0, vmax=1.0)#np.sum(test_input[0, 16, :, :, :], axis=2))
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title("predicted followup")
        plt.imshow(to_tar[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title("ground truth")
        plt.imshow(ground_truth[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title("predicted mask")
        plt.imshow(to_tar[0, 16, :, :, 1], cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title("ground truth mask")
        plt.imshow(ground_truth_mask[0, 16, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')

    else:

        if not use_avg_only:
            plt.subplot(2, 2, 1)
            plt.title("compressed pwi")
            imm = np.moveaxis(np.array([test_input[0, 16, :, :, 0], test_input[0, 16, :, :, 1], test_input[0, 16, :, :, 3]]), 0, -1)
            plt.imshow(imm)#test_input[0, 16, :, :, 0], vmin=-1.0, vmax=1.0)#np.sum(test_input[0, 16, :, :, :], axis=2))
            plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("avg")
        plt.imshow(test_input[0, 16, :, :, -1], cmap='gray', vmin=-1.0, vmax=1.0)#np.sum(test_input[0, 16, :, :, :], axis=2))
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title("predicted followup")
        plt.imshow(to_tar[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title("ground truth")
        plt.imshow(ground_truth[0, 16, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')


    plt.savefig(folder_prepend + label + '.jpg', dpi=300)
    plt.close()


def alt_dice(im1, im2):
    im1 = im1[0, :, :, :, 0]
    im2 = im2[0, :, :, :, 0]
    #im1 = np.where(im1>0.5, 1.0, 0.0)
    #im2 = np.where(im2>0.5, 1.0, 0.0)
    intersection = im1 * im2
    return 1 - ((2.0 * tf.math.reduce_sum(intersection)) / (tf.math.reduce_sum(im1) + tf.math.reduce_sum(im2) + 0.00000000001))



@tf.function
def train_step_followup(x, gt, lesion):

    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    #seg_real = seg_unet(gt)

    #seg_fake = seg_unet(generator(x))

    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y. FLAIR to NCCT. Discriminator Y.
        # Generator F translates Y -> X. NCCT to FLAIR. Discriminator X.
        if pred_mask:

            y = generator(x, training=True)

            disc_real = discriminator(tf.concat([x, gt, lesion], axis=4), training=True)

            disc_fake = discriminator(tf.concat([x, y], axis=4), training=True)

            gen_loss = generator_loss(disc_fake)

            disc_loss = discriminator_loss(disc_real, disc_fake)

            total_gen_loss = (LAMBDA * l1_loss(y[:, :, :, :, 0, tf.newaxis], gt)) + (LAMBDA_M * alt_dice(y[:, :, :, :, 1, tf.newaxis], lesion)) + (LAMBDA_C * gen_loss)
        
        else:

            y = generator(x, training=True)

            disc_real = discriminator(tf.concat([x, gt], axis=4), training=True)

            disc_fake = discriminator(tf.concat([x, y], axis=4), training=True)

            gen_loss = generator_loss(disc_fake)

            disc_loss = discriminator_loss(disc_real, disc_fake)

            total_gen_loss = (LAMBDA * l1_loss(y, gt)) + (LAMBDA_C * gen_loss)

        
            #asym_unified_focal_loss(seg_real, seg_fake))

    generator_gradients = tape.gradient(total_gen_loss, 
            generator.trainable_variables)

    discriminator_gradients = tape.gradient(disc_loss,
            discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
            generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
            discriminator.trainable_variables))

    return total_gen_loss, disc_loss


def evaluate_loss():

    accum_y = []

    for kk in range(source_test_followup.shape[0]):

        if use_pmaps:
            image_x = getparams(source_test_params, kk)
        else:
            image_x = encode_vol(getimg(source_test_pwi[kk]))

        image_avg = getimg(source_test_avg[kk])
        gt = getimg(source_test_followup[kk])
        lesion = getimg(source_test_lesions[kk])

        if use_avg:
            x = np.concatenate((image_x, image_avg), axis=4, dtype="float32")
        else:
            x = image_x

        if use_avg_only:
            x = image_avg

        y = generator(x, training=True)

        #seg_real = seg_unet(gt, training=False)

        #seg_fake = seg_unet(y, training=False)
        if pred_mask:

            disc_real = discriminator(tf.concat([x, gt, lesion], axis=4), training=True)

            disc_fake = discriminator(tf.concat([x, y], axis=4), training=True)

            gen_loss = generator_loss(disc_fake)

            disc_loss = discriminator_loss(disc_real, disc_fake)\

            #structural_loss = 1-NGF(gt, y)

            # calculate the loss
            total_gen_loss = (LAMBDA * l1_loss(y, gt)) + (LAMBDA_M * alt_dice(y[:, :, :, :, 1, tf.newaxis], lesion)) + (LAMBDA_C * gen_loss) #+ (LAMBDA_D * structural_loss)#+ (LAMBDA_B * 
                #asym_unified_focal_loss(seg_real, seg_fake)) 
        
        else:

            disc_real = discriminator(tf.concat([x, gt], axis=4), training=True)

            disc_fake = discriminator(tf.concat([x, y], axis=4), training=True)

            gen_loss = generator_loss(disc_fake)

            disc_loss = discriminator_loss(disc_real, disc_fake)\

            #structural_loss = 1-NGF(gt, y)

            # calculate the loss
            total_gen_loss = (LAMBDA * l1_loss(y, gt)) + (LAMBDA_C * gen_loss) #+ (LAMBDA_D * structural_loss)#+ (LAMBDA_B * 
            


        accum_y.append(total_gen_loss)


    accum_y = np.array(accum_y)
    accum_y = accum_y.mean()

    return accum_y


# Define augmentation functions

def random_rotation(im1, im2):

    #dims => tuple with two axes define plane of rotation
    dims = (1, 2)
    # extract
    im1 = im1[0, :, :, :, 0]
    im2 = im2[0, :, :, :, :]

    # process
    ran = random.randint(-90, 90)
    im1_rot = ndimage.rotate(im1, ran, reshape=False, order=1, axes=dims, mode="nearest")
    im2_rot = ndimage.rotate(im2, ran, reshape=False, order=0, axes=dims, mode="nearest")

    # return
    im1_rot = im1_rot[np.newaxis, :, :, :, np.newaxis]
    im2_rot = im2_rot[np.newaxis, :, :, :, :]
    
    return [im1_rot, im2_rot]


def random_scaling(im1, im2):

    # extract
    im1 = im1[0, :, :, :, 0]
    im2 = im2[0, :, :, :, :]

    bck = im1[0,0,0]
    # process
    ran = random.uniform(-0.3, 0.3)
    #print(ran)

    im1 = scipy.ndimage.interpolation.zoom(im1, [1, 1+ran, 1+ran], order=1, mode='nearest')
    im2 = scipy.ndimage.interpolation.zoom(im2, [1, 1+ran, 1+ran, 1], order=0, mode='nearest')

    #print(im1.shape)
    #print(im2.shape)

    im1_sca = np.full(np.array([32, 224, 160]), bck, np.float32)
    im2_sca = np.zeros(np.array([32, 224, 160, 2]), np.float32)
    im2_sca[:, :, :, 1] = np.full(np.array([32, 224, 160]), 1.0, np.float32)

    dx = 224
    dy = 160

    xx = np.floor(im1.shape[0]/2)
    yy = np.floor(im1.shape[1]/2)
    zz = np.floor(im1.shape[2]/2)

    if (ran<0):
        im1_sca[int(16-xx):int(16-xx+im1.shape[0]), int(int(dx/2)-yy):int(int(dx/2)-yy+im1.shape[1]), int(int(dy/2)-zz):int(int(dy/2)-zz+im1.shape[2])] = im1    
        im2_sca[int(16-xx):int(16-xx+im2.shape[0]), int(int(dx/2)-yy):int(int(dx/2)-yy+im2.shape[1]), int(int(dy/2)-zz):int(int(dy/2)-zz+im2.shape[2]), :]  = im2
    else:
        im1_sca = im1[int(xx-16):int(xx-16+32), int(yy-int(dx/2)):int(yy-int(dx/2)+dx), int(zz-int(dy/2)):int(zz-int(dy/2)+dy)]    
        im2_sca = im2[int(xx-16):int(xx-16+32), int(yy-int(dx/2)):int(yy-int(dx/2)+dx), int(zz-int(dy/2)):int(zz-int(dy/2)+dy), :]   

    # return
    im1_sca = im1_sca[np.newaxis, :, :, :, np.newaxis]
    im2_sca = im2_sca[np.newaxis, :, :, :, :]

    #print(im1_sca.shape)
    #print(im2_sca.shape)
    
    return [im1_sca.astype('float32'), im2_sca.astype('float32')]


def mirror(im1, im2, im3, im4):

    direction = 3#random.randint(1, 3)

    # extract
    im1 = im1[0, :, :, :, :]
    im2 = im2[0, :, :, :, :]
    im3 = im3[0, :, :, :, :]
    im4 = im4[0, :, :, :, :]

    if(direction==1):
        # process
        im1_flip = im1[::-1, :, :, :]
        im2_flip = im2[::-1, :, :, :]
    elif(direction==2):
        # process
        im1_flip = im1[:, ::-1, :, :]
        im2_flip = im2[:, ::-1, :, :]
    else:
        # process
        im1_flip = im1[:, :, ::-1, :]
        im2_flip = im2[:, :, ::-1, :]
        im3_flip = im3[:, :, ::-1, :]
        im4_flip = im4[:, :, ::-1, :]

    # return
    im1_flip = im1_flip[np.newaxis, :, :, :, :]
    im2_flip = im2_flip[np.newaxis, :, :, :, :]
    im3_flip = im3_flip[np.newaxis, :, :, :, :]
    im4_flip = im4_flip[np.newaxis, :, :, :, :]
    
    return [im1_flip, im2_flip, im3_flip, im4_flip]


# Start elastic augmentation generator.

elastic_gen = ea.WarpEngine(0)


loss = []
loss_disc = []
val_loss = []
BATCH_SIZE = 3



for epoch in range(EPOCHS):
    start = time.time()
    daloss = []
    daloss_disc = []
    src_params_shuffled = [None]*4
    
    src_idxs = np.array(range(source_images_pwi.shape[0]))
    np.random.shuffle(src_idxs)
    src_followup_shuffled = source_images_followup[src_idxs]
    src_pwi_shuffled = source_images_pwi[src_idxs]
    src_params_shuffled[0] = source_images_params[0][src_idxs] # cbf
    src_params_shuffled[1] = source_images_params[1][src_idxs] # cbv
    src_params_shuffled[2] = source_images_params[2][src_idxs] # mtt
    src_params_shuffled[3] = source_images_params[3][src_idxs] # tmax
    src_avg_shuffled = source_images_avg[src_idxs]
    src_lesion_shuffled = source_images_lesions[src_idxs]


    batches = int(np.floor(len(source_images_followup)/BATCH_SIZE))
    last_batch = len(source_images_followup)-(batches*BATCH_SIZE)
    for pr in range(batches):#len(source_images_followup)): 

        offset_batch = pr*BATCH_SIZE
        image_x = []
        image_avg = []
        image_y = []
        image_lesion = []

        for prr in range(BATCH_SIZE):
            if use_pmaps:
                image_x.append(getparams(src_params_shuffled, offset_batch+prr))
            else:
                image_x.append(encode_vol(getimg(src_pwi_shuffled[offset_batch+prr])))
            image_avg.append(getimg(src_avg_shuffled[offset_batch+prr]))
            image_y.append(getimg(src_followup_shuffled[offset_batch+prr]))
            image_lesion.append(getimg(src_lesion_shuffled[offset_batch+prr]))
            #image_y_label = getlabel()


        image_x = np.concatenate(tuple(image_x), axis=0, dtype="float32")
        image_avg = np.concatenate(tuple(image_avg), axis=0, dtype="float32")
        image_y = np.concatenate(tuple(image_y), axis=0, dtype="float32")
        image_lesion = np.concatenate(tuple(image_lesion), axis=0, dtype="float32")

        #print(image_x.shape)
        #print(image_avg.shape)
        #print(image_y.shape)
        #print(image_lesion.shape)
        #NGF(generator(np.concatenate((image_x, image_avg), axis=4, dtype="float32")), image_y)

        if use_avg:
            dalossy, dalossy_disc = train_step_followup(np.concatenate((image_x, image_avg), axis=4, dtype="float32"), image_y, image_lesion)
        else:
            if use_avg_only:
                dalossy, dalossy_disc = train_step_followup(image_avg, image_y, image_lesion)
            else:
                dalossy, dalossy_disc = train_step_followup(image_x, image_y, image_lesion)

    
        daloss.append(dalossy)
        daloss_disc.append(dalossy_disc)        
        
        # Data augmentation on the fly
        #if pr == 1:
            #NGF(generator(np.concatenate((image_x, image_avg), axis=4, dtype="float32")), image_y)

        #GC_show(generator(np.concatenate((image_x, image_avg), axis=4, dtype="float32")), image_y)

        # Flip random
        daflip = mirror(image_x, image_y, image_avg, image_lesion)
        #np.concatenate((daflip[0], daflip[2]), axis=4, dtype="float32")
        if use_avg:
            dalossy, dalossy_disc = train_step_followup(np.concatenate((daflip[0], daflip[2]), axis=4, dtype="float32"), daflip[1], daflip[3])

        else:
            if use_avg_only:
                dalossy, dalossy_disc = train_step_followup(daflip[2], daflip[1], daflip[3])
            else:
                dalossy, dalossy_disc = train_step_followup(daflip[0], daflip[1], daflip[3])



        daloss.append(dalossy)
        daloss_disc.append(dalossy_disc)

        """
        if pr == 1:
            generate_images(generator, daflip[0], daflip[1], "flip_x_"+str(epoch))
        """


        print("dataset "+str(pr+1)+"/"+str(len(src_followup_shuffled))+" loss:"+str(np.array(daloss).mean()))


    if pred_mask:
        generate_images(generator, sample_src_con, sample_src_followup, str(dafold)+"/"+"train_pred_e"+str(epoch), sample_src_lesions)
        generate_images(generator, val_sample_src_con, val_sample_src_followup, str(dafold)+"/"+"val_pred_e"+str(epoch), val_sample_src_lesions)
    else:
        generate_images(generator, sample_src_con, sample_src_followup, str(dafold)+"/"+"train_pred_e"+str(epoch))
        generate_images(generator, val_sample_src_con, val_sample_src_followup, str(dafold)+"/"+"val_pred_e"+str(epoch))

    #print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    daloss = np.array(daloss)
    daloss = daloss.mean()
    daloss_disc = np.array(daloss_disc)
    daloss_disc = daloss_disc.mean()
    daloss_val = evaluate_loss()
    print('Followup prediction loss: ' + str(daloss) + ' - disc loss: ' + str(daloss_disc) + ' - val loss: ' + str(daloss_val))

    
    loss.append(daloss)
    val_loss.append(daloss_val)
    loss_disc.append(daloss_disc)


    if (epoch+1)%15 == 0:

        epochss = range(epoch+1)


        plt.figure()
        plt.plot(epochss, loss, 'c', label='Prediction loss', linewidth=0.5)
        plt.plot(epochss, val_loss, 'm', label='Validation loss', linewidth=0.5)
        plt.plot(epochss, loss_disc, 'b', label='Discriminator loss', linewidth=0.5)
        plt.title('loss', fontsize=10)
        plt.legend()


generator.save_weights(folder_prepend +"/"+str(dafold)+"/"+ 'generator.h5')
generator.save(folder_prepend +"/"+str(dafold)+"/"+ 'generator')

discriminator.save_weights(folder_prepend +"/"+str(dafold)+"/"+ 'discriminator.h5')
discriminator.save(folder_prepend +"/"+str(dafold)+"/"+ 'discriminator')

# Run the trained model on the test dataset
for inp in range(source_test_followup.shape[0]):

    if use_pmaps:
        if use_avg:
            zip_pwi = np.concatenate((getparams(source_test_params, inp), getimg(source_test_avg[inp])), axis=4, dtype="float32")
        else:
            zip_pwi = getparams(source_test_params, inp)
    else:
        if use_avg:
            zip_pwi = np.concatenate((encode_vol(getimg(source_test_pwi[inp])), getimg(source_test_avg[inp])), axis=4, dtype="float32")
        else:
            zip_pwi = encode_vol(getimg(source_test_pwi[inp]))

    if use_avg_only:
        zip_pwi = getimg(source_test_avg[inp])

    fp = getimg(source_test_followup[inp])
    fp_mask = getimg(source_test_lesions[inp])

    if pred_mask:
        generate_images(generator, zip_pwi, fp, "test"+"/"+dafold+"/"+"test_followup_"+str(inp), fp_mask)
    else:
        generate_images(generator, zip_pwi, fp, "test"+"/"+dafold+"/"+"test_followup_"+str(inp))


print(loss)
print(val_loss)

epochss = range(EPOCHS)

plt.figure()
plt.plot(epochss, loss, 'c', label='Prediction loss', linewidth=0.5)
plt.plot(epochss, val_loss, 'm', label='Validation loss', linewidth=0.5)
plt.plot(epochss, loss_disc, 'b', label='Discriminator loss', linewidth=0.5)
plt.title('loss', fontsize=10)
plt.legend()
plt.savefig(folder_prepend +"/"+dafold+"/"+ 'followup_loss_curve.png')