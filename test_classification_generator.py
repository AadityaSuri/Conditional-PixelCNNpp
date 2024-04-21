'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
from utils import *
from model import *
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import numpy as np

NUM_CLASSES = len(my_bidict)


def get_label(model, model_input, device):
    class_losses = []
    for _, i in my_bidict.items():
        dummy_label = torch.full((len(model_input),), i, dtype=torch.int64).to(device)
        dummy_output = model(model_input, class_label=dummy_label)
        class_loss = discretized_mix_logistic_loss(model_input, dummy_output, 'classification')
        class_losses.append(class_loss)

    class_losses = torch.stack(class_losses)
    _, predicted_class = class_losses.min(dim=0)

    return predicted_class, class_losses



def classifier(model, data_loader, device, test_res_csv, logits_csv):
    model.eval()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, _, img_path = item
        model_input = model_input.to(device)
        answer, class_losses = get_label(model, model_input, device)
        npy_rows = torch.transpose(class_losses, 0, 1)

        for row in npy_rows:
            row_str = ','.join(map(str, row.tolist()))
            logits_csv.write(row_str + '\n')

        for i, img_path_ in enumerate(img_path):
            img_name = img_path_.split('/')[-1]
            strcsv = f'{img_name},{answer[i]}\n'
            test_res_csv.write(strcsv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=8, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    parser.add_argument('-f', '--fid', type=float,
                        default=455, help='FID score')

    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(TestClassificationDataset(root_dir=args.data_dir,
                                                            mode=args.mode,
                                                            transform=ds_transforms),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             **kwargs)



    model = PixelCNN(nr_resnet=2, nr_filters=160, input_channels=3, nr_logistic_mix=10)
    model = model.to(device)

    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth', map_location=device)['model_state_dict'])
    model.eval()
    print('model parameters loaded')

    f = open('test_classification_results.csv', 'w')
    logits_csv = open('test_logits.csv', 'w')

    f.write('id,label\n')
    classifier(model=model, data_loader=dataloader, device=device, test_res_csv=f, logits_csv=logits_csv)
    f.write(f'fid,{args.fid}\n')


    f.close()
    logits_csv.close()

    file_path = 'test_logits.csv'  # Replace with the path to your CSV file
    data = np.genfromtxt(file_path, delimiter=',')  # You can adjust the delimiter if needed
    np.save('test_logits.npy', data)

    loaded_data = np.load('test_logits.npy')
    print("Data shape:", loaded_data.shape)
    print(loaded_data)

