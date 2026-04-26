import torch
import time
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for SDv1.4',
                    description = 'Finetuning stable-diffusion to erase the concepts')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    return parser




if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()



    # device = torch.device("cuda:0")

    # Allocate ~8GB (float32 = 4 bytes)
    num_elements = (8 * 1024**3) // 4
    tensor = torch.empty(num_elements, dtype=torch.float32).to(args.device)

    # print("Reserved ~8GB of GPU memory. Holding...")

    # Keep process alive
    while True:
        time.sleep(1000)