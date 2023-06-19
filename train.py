from packages import *
from utils import *

def train(
    new_datasets=False
):
    # make train & val
    make_train_val(root_path="datasets/car", run_ok=new_datasets)
    # build from YAML and transfer weights
    model = YOLO('yolov5s6.yaml').load('pretrained/yolov5s6.pt')

    # # Train the model
    model.train(cfg="cfg/custom.yaml")


if __name__ == "__main__":
    train(new_datasets=True)