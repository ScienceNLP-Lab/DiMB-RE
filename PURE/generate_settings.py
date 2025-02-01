import argparse
import sys
import os


# create directory
def make_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Numbering the version of output
    existing_folders = []
    for d in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("EXP_"):
            existing_folders.append(int(d.split("_")[1]))

    num_version = 0
    if existing_folders:
        num_version = max(existing_folders) + 1

    model_output_dir = os.path.join(output_dir, f"EXP_{num_version}") 
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)        
    print(f"## {model_output_dir} is created for Experiment outputs ##")
    

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', help='Directory for experiments', type=str, default='experiments')
    args = parser.parse_args(arguments)

    make_output_dir(args.experiment_dir)

if __name__ == '__main__':
    main(sys.argv[1:])