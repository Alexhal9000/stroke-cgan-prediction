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
import pickle
import sys

import os
import time
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

# load scans NCCT and FLAIR

LAMBDA = 1 # 4
#LAMBDA_B = 0.1 # 0.4   last 0.05

EPOCHS = 40

learning_rate = 2e-4 # last: 2e-5
scale_factor = 1
target_scale = 192
size_offset = 0



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
    return np.load(name)[np.newaxis, ..., np.newaxis].astype("float32")


def test_encoder(vol):
    print(vol.shape)
    out_vol = []
    for n in range(32):
        out_vol.append(np.ravel(vol[:, :, :, :, n], order='C'))

    return np.array(out_vol)


def test_latent(vol):
    print(vol.shape)
    out_vol = np.reshape(vol[:, n], (1, 32, 448, 320, 1), order='C')
    return np.array(out_vol)



names, names_extracted = extract_names("./pre_processed/npy/interpolated_pwi/*","pwi/",".npy")


treatment = "IVT"



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

folder_prepend = "./eval_1/autoencoder_4/"+str(dafold)+"/"

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

"""
source_images = []
for name in names:
    source_images.append(np.load(name)) # X
    print_ram()

source_images = np.array(source_images)
print(source_images.shape)
"""

np.random.seed(10)
random.seed(10)

src_len = len(names)
src_order = np.arange(src_len)

np.random.shuffle(src_order)
print(src_order)


# do training validation testing split
# ------------------------------------
# -------- Load pwi ------

names_pwi = np.array(names)
source_images = names_pwi[src_order]


# do training validation testing split
"""
source_val_pwi = source_images_pwi[(math.floor(src_len*0.7)):(math.floor(src_len*0.85))]

source_test_pwi = source_images_pwi[(math.floor(src_len*0.85)):src_len]

source_images_pwi = source_images_pwi[0:(math.floor(src_len*0.7))]
"""

source_test = np.array(["./pre_processed/npy/interpolated_pwi/" + s + ".npy" for s in nams_test])
source_images = np.array(["./pre_processed/npy/interpolated_pwi/" + s + ".npy" for s in nams_tr])



print("Source shape pwi:")
print(source_images.shape)

#print("Validation source shape pwi:")
#print(source_val_pwi.shape)

print("Test source shape pwi:")
print(source_test.shape)


sample_src = getimg(source_images[5])

val_sample_src = getimg(source_test[5])

print("Source sample shape pwi:")
print(sample_src.shape)



BUFFER_SIZE = 10
BATCH_SIZE = 1


OUTPUT_CHANNELS = 4

encoder = pix2pix.encoder(OUTPUT_CHANNELS, norm_type='instancenorm')
print(encoder.summary())
decoder = pix2pix.decoder(1, OUTPUT_CHANNELS, norm_type='instancenorm')
print(decoder.summary())

print(sample_src[:, 16, :, :, :, :].shape)
to_tar = encoder(sample_src[:, 16, :, :, :, :])
print(to_tar.shape)
to_reconstruct = decoder(to_tar)

plt.subplot(1, 3, 1)
plt.title("2D+t")
plt.imshow(sample_src[0, 16, :, :, 16, 0], cmap='gray', vmin=-1.0, vmax=1.0)

plt.subplot(1, 3, 2)
plt.title("to 2D")
plt.imshow(to_tar[0, :, :, 0:3, 0], vmin=-1.0, vmax=1.0)

plt.subplot(1, 3, 3)
plt.title("rec 2D+t")
plt.imshow(to_reconstruct[0, :, :, 16, 0], cmap='gray', vmin=-1.0, vmax=1.0)

#plt.show()
plt.savefig(folder_prepend + "initial_state_gen.png")
plt.close()


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):

    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def auto_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss



encoder_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999)
decoder_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999)


def generate_images(model1, model2, test_input, label):

    to_tar = model1(test_input)

    to_reconstruct = model2(to_tar)

    plt.subplot(1, 3, 1)
    plt.title("2D+t")
    plt.imshow(test_input[0, :, :, 16, 0], cmap='gray', vmin=-1.0, vmax=1.0)

    plt.subplot(1, 3, 2)
    plt.title("to 2D")
    plt.imshow(to_tar[0, :, :, 0:3, 0], vmin=-1.0, vmax=1.0)

    plt.subplot(1, 3, 3)
    plt.title("rec 2D+t")
    plt.imshow(to_reconstruct[0, :, :, 16, 0], cmap='gray', vmin=-1.0, vmax=1.0)

    #plt.show()
    plt.axis('off')

    #plt.show()
    plt.savefig(folder_prepend + label + '.png')
    plt.close()


# sigma 1
kernel_gauss = np.array([ 
    [ [ [0.07] ],[ [0.12] ],[ [0.07] ] ],
    [ [ [0.12] ],[ [0.19] ],[ [0.12] ] ],
    [ [ [0.07] ],[ [0.12] ],[ [0.07] ] ],
    ])
lesion_kernel_gaussian = tf.constant(kernel_gauss, dtype=tf.float32)


@tf.function
def train_step(real_x):

    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y. FLAIR to NCCT. Discriminator Y.
        # Generator F translates Y -> X. NCCT to FLAIR. Discriminator X.

        y = encoder(real_x, training=True)
        fake_x = decoder(y, training=True)

        # calculate the loss
        total_loss = auto_loss(real_x, fake_x)


    # Calculate the gradients for generator and discriminator
    encoder_gradients = tape.gradient(total_loss,
                                        encoder.trainable_variables)
    
    decoder_gradients = tape.gradient(total_loss,
                                            decoder.trainable_variables)

    # Apply the gradients to the optimizer
    encoder_optimizer.apply_gradients(zip(encoder_gradients,
                                            encoder.trainable_variables))

    decoder_optimizer.apply_gradients(zip(decoder_gradients,
                                                decoder.trainable_variables))

    return total_loss



def evaluate_loss():

    accum_y = []

    for kk in range(source_test.shape[0]):

        real_x = source_test[kk]

        for aslice in range(sample_src.shape[1]):

            xx = getimg(real_x)[:, aslice, :, :, :, :]
            y = encoder(xx, training=False)
            fake_x = decoder(y, training=False)

            # calculate the loss
            total_loss = auto_loss(xx, fake_x)

            
            accum_y.append(total_loss)


    accum_y = np.array(accum_y)
    accum_y = accum_y.mean()

    return accum_y


loss_auto = []
val_loss_auto = []




"""
# Start elastic augmentation generator.

elastic_gen = ea.WarpEngine(0)
"""

# Define augmentation functions

def random_rotation(im1):

    #dims => tuple with two axes define plane of rotation
    dims = (0, 1)
    # extract
    im1 = im1[0, :, :, :, 0]

    # process
    ran = random.randint(20, 340)
    im1_rot = ndimage.rotate(im1, ran, reshape=False, order=1, axes=dims, mode="nearest")

    # return
    im1_rot = im1_rot[np.newaxis, :, :, :, np.newaxis]
    
    return im1_rot

def mirror_horizontal(im1):

    # extract
    im1 = im1[0, :, :, :, 0]

    # process
    im1_flip = im1[:, ::-1, :]

    # return
    im1_flip = im1_flip[np.newaxis, :, :, :, np.newaxis]
    
    return im1_flip

"""
def mirror_vertical(im1, im2):

    # extract
    im1 = im1[0, :, :, 0]
    im2 = im2[0, :, :, 0]

    # process
    im1_flip = np.flipud(im1)
    im2_flip = np.flipud(im2)

    # return
    im1_flip = im1_flip[np.newaxis, :, :, np.newaxis]
    im2_flip = im2_flip[np.newaxis, :, :, np.newaxis]

    return im1_flip, im2_flip
"""

# Start training.

for epoch in range(EPOCHS):
    start = time.time()
    daloss = []
    
    src_idxs = np.array(range(source_images.shape[0]))
    np.random.shuffle(src_idxs)
    src_shuffled = source_images[src_idxs]

    for pr in range(len(src_shuffled)):#BUFFER_SIZE): 

        image_x = src_shuffled[pr]

        for aslice in range(sample_src.shape[1]):

            dalossy = train_step(getimg(image_x)[:, aslice, :, :, :, :])
    
        daloss.append(dalossy)
        

        
        # Data augmentation on the fly
        # Rotation
        """
        for aslice in range(sample_src.shape[1]):

            dalossy = train_step(random_rotation(getimg(image_x)[:, aslice, :, :, :, :]))
    
        daloss.append(dalossy)

        """

        """
        #if pr == 6:
        #    generate_images(generator_g, image_x, "rotate_x_"+str(epoch))

        """

        # Flip horizontal
        """
        for aslice in range(sample_src.shape[1]):

            dalossy = train_step(mirror_horizontal(getimg(image_x)[:, aslice, :, :, :, :]))
    
        daloss.append(dalossy)
        """

        """
        #if pr == 1:
         #   generate_images(generator_g, image_x, "fliplr_x_"+str(epoch))

        """
        """
        # Flip vertical
        image_x, image_y = mirror_vertical(src_shuffled[pr], tar_shuffled[pr])

        dlossy, glossg = train_step(image_x, image_y)
        
        ddlossy.append(dlossy)
        dglossg.append(glossg)

        """
        #if pr == 1:
        #    generate_images(generator_g, image_x, "flipud_x_"+str(epoch))

        """

        # Elastic deformations
        image_x = elastic_gen.generate(src_shuffled[pr], 1)
        image_y = elastic_gen.generate(tar_shuffled[pr], 1)        

        dlossy, glossg = train_step(image_x, image_y)
        
        ddlossy.append(dlossy)
        dglossg.append(glossg)

        """
        #if pr == 6:
        #    generate_images(generator_g, image_x, "elastic_x_"+str(epoch))
        """



        """
        print("dataset "+str(pr+1)+"/"+str(len(src_shuffled))+" loss:"+str(np.array(daloss).mean()))


    generate_images(encoder, decoder, sample_src[:, 16, :, :, :, :], "train_auto_e"+str(epoch))
    #generate_images(encoder, decoder, random_rotation(mirror_horizontal(sample_src[:, 16, :, :, :, :])), "rotate_auto_e"+str(epoch))
    generate_images(encoder, decoder, val_sample_src[:, 16, :, :, :, :], "val_auto_e"+str(epoch))


    #GC_display(list(mr_pair.as_numpy_iterator())[23][0], generator_g(list(mr_pair.as_numpy_iterator())[23][0]), list(mr_pair.as_numpy_iterator())[23][1], str(epoch), "FLAIR_GND", "NCCT_GND")

    """
    if (epoch + 1) % 25 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    """

    #print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    daloss = np.array(daloss)
    daloss = daloss.mean()
    daloss_val = evaluate_loss()
    print('Autoencoder loss: ' + str(daloss) + ' - val loss: ' + str(daloss_val))

    
    loss_auto.append(daloss)
    val_loss_auto.append(daloss_val)


    if (epoch+1)%15 == 0:

        epochss = range(epoch+1)


        plt.figure()
        plt.plot(epochss, loss_auto, 'co', label='Autoencoder loss', linewidth=0.5)
        plt.plot(epochss, val_loss_auto, 'm', label='Validation loss', linewidth=0.5)
        plt.title('loss', fontsize=10)
        plt.legend()
        plt.savefig(folder_prepend + 'loss_curve.png')

        """
        generator_g.save_weights(folder_prepend + 'checkpoints/generator_mr_to_ct_'+str(epoch+1)+'.h5')
        generator_f.save_weights(folder_prepend + 'checkpoints/generator_ct_to_mr_'+str(epoch+1)+'.h5')
        discriminator_x.save_weights(folder_prepend + 'checkpoints/discriminator_mr_to_ct_'+str(epoch+1)+'.h5')
        discriminator_y.save_weights(folder_prepend + 'checkpoints/discriminator_ct_to_mr_'+str(epoch+1)+'.h5')
        """


def decode_vol(image, decoder, encoder):
    out = np.empty((32, 224, 160, 32))
    for a in range(32):
        #print(encoder(sample_src_pwi[:, a, :, :, :, :]).shape)
        out[a, :, :, :] = decoder(encoder(image[:, a, :, :, :, :]))[0, :, :, 0]
    #print("out shape: "+out.shape)
    out = out[np.newaxis, ..., np.newaxis]
    return out.astype("float32")


# Run the trained model on the test dataset
test_score = []
for inp in range(source_test.shape[0]):
    tr = source_test[inp]
    rec = decode_vol(getimg(tr), decoder, encoder)
    print(rec.shape)
    test_score.append(np.mean(np.absolute(getimg(tr) - rec)))
    generate_images(encoder, decoder, getimg(tr)[:, 16, :, :, :, :], "test_auto/test_auto_"+str(inp))

print(str(np.mean(np.array(test_score))))
print(str(np.std(np.array(test_score))))

print(loss_auto)
print(val_loss_auto)

epochss = range(EPOCHS)

plt.figure()
plt.plot(epochss, loss_auto, 'bo', label='Autoencoder loss', linewidth=0.5)
plt.plot(epochss, val_loss_auto, 'r', label='Validation loss', linewidth=0.5)
plt.title(str(np.mean(np.array(test_score))) + ' ' + str(np.std(np.array(test_score))), fontsize=10)
plt.legend()
plt.savefig(folder_prepend + 'loss_curve.png')

encoder.save_weights(folder_prepend + 'encoder.h5')
decoder.save_weights(folder_prepend + 'decoder.h5')

encoder.save(folder_prepend + 'encoder')
decoder.save(folder_prepend + 'decoder')


