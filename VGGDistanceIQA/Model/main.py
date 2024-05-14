from model import Multiscale, extract_features
import torch
import numpy as np
import argparse



def main(args):


    model = Multiscale()
    model.load_model(args.m)
    model.eval()

    layer_names = ['conv1_1','conv2_2','conv3_3','conv4_3','conv5_3']
    assessed_features = extract_features(args.t,  layer_names)
    reference_features = extract_features(args.r, layer_names)

    diff = assessed_features - reference_features
    diff_tensor = torch.tensor(diff, dtype=torch.float32).unsqueeze(0)

    output = model(diff_tensor)
    prediction = torch.argmax(output, dim=1)


    print("+-------result------+")
    print("|      Score:", prediction.item(),"    |")
    print("+-------Legend------+")
    print("|   0: Bad  9:Best  |")
    print("+-------------------+")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, required=True, help="trained model file")
    parser.add_argument("--t", type=str, required=True, help="file to assessed image")
    parser.add_argument("--r", type=str, required=True, help="file to reference image")
    args = parser.parse_args()
    main(args)