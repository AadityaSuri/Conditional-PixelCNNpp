'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify the remaining code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
2. Modify how you call your sample function(line 31)
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import *
from dataset import *
import os
import torch
# You should modify this sample function to get the generated images from your model
# This function should save the generated images to the gen_data_dir, which is fixed as 'samples'
# Begin of your code
sample_op = lambda x : sample_from_discretized_mix_logistic(x, 10)
def sample_(model, gen_data_dir, sample_batch_size = 100, obs = (3,32,32), sample_op = sample_op):
    sample_t = sample(model, sample_batch_size, obs, sample_op)
    sample_t = rescaling_inv(sample_t)
    save_images(sample_t, os.path.join(gen_data_dir), label='test')

# End of your code

if __name__ == "__main__":
    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    BATCH_SIZE=192  # I changed this to 192 because I was getting the FID error for generating less than 48 images per
                    # class = 192.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    #Begin of your code
    #Load your model and generate images in the gen_data_dir
    model = PixelCNN(nr_resnet=2, nr_filters=160, input_channels=3, nr_logistic_mix=10)
    checkpoint = torch.load("models/conditional_pixelcnn.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model = model.eval()
    sample_(model=model, gen_data_dir=gen_data_dir, sample_batch_size=BATCH_SIZE)
    #End of your code
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))

    print("Average fid score: {}".format(fid_score))