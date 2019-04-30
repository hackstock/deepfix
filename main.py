import argparse
import sys
import os
import shutil
from core.faces.loader import Recognizer
from core.models.convnet import ConvNet

def run(**kwargs):
    print("deepfix is running")

def capture(**kwargs):
    path = kwargs["path"]
    mode = kwargs["mode"]
    pos = kwargs["pos"]
    neg = kwargs["neg"]
    size = kwargs["size"]

    images_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "images"))
    

    if mode == "reset" and os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    elif mode == "append" and not os.path.exists(path):
        os.mkdir(path)
        
    files_count = len([f for f in os.listdir(images_dir) if f.startswith(pos)])
    rec = Recognizer(src=0, haar_path="./haarcascade_frontalface.xml",files_count=files_count, pos_prefix=pos, neg_prefix=neg, size=size)
    rec.run()

def train(**kwargs):
    known_models = ["convnet", "logisticreg", "xception"]
    model = kwargs["model"]
    pos = kwargs["pos"]
    neg = kwargs["neg"]
    train_size = int(kwargs["training"]) + 1
    validation_size = int(kwargs["validation"])
    test_size = int(kwargs["testing"])


    if not model in known_models:
        print("{model}: model not found. --model must be one of {models}".format(model=model, models=known_models))
        sys.exit(-1)

    network = None
    if model == "convnet":
        network = ConvNet()
        print(network.model.summary())

    images_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "images"))
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "data"))
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    os.mkdir(data_dir)

    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    train_pos_dir = os.path.join(train_dir, pos)
    train_neg_dir = os.path.join(train_dir, neg)
    validation_pos_dir = os.path.join(validation_dir, pos)
    validation_neg_dir = os.path.join(validation_dir, neg)
    test_pos_dir = os.path.join(test_dir, pos)
    test_neg_dir = os.path.join(test_dir, neg)

    os.mkdir(train_pos_dir)
    os.mkdir(train_neg_dir)
    os.mkdir(validation_pos_dir)
    os.mkdir(validation_neg_dir)
    os.mkdir(test_pos_dir)
    os.mkdir(test_neg_dir)

    fnames = ["{}.{}.png".format(pos, i) for i in range(1, train_size)]
    for fname in fnames:
        src = os.path.join(images_dir, fname)
        dest = os.path.join(train_pos_dir, fname)
        shutil.copyfile(src, dest)

    fnames = ["{}.{}.png".format(neg, i) for i in range(1, train_size)]
    for fname in fnames:
        src = os.path.join(images_dir, fname)
        dest = os.path.join(train_neg_dir, fname)
        shutil.copyfile(src, dest)

    fnames = ["{}.{}.png".format(pos, i) for i in range(train_size, train_size + validation_size)]
    for fname in fnames:
        src = os.path.join(images_dir, fname)
        dest = os.path.join(validation_pos_dir, fname)
        shutil.copyfile(src, dest)

    fnames = ["{}.{}.png".format(neg, i) for i in range(train_size, train_size + validation_size)]
    for fname in fnames:
        src = os.path.join(images_dir, fname)
        dest = os.path.join(validation_neg_dir, fname)
        shutil.copyfile(src, dest)

    fnames = ["{}.{}.png".format(pos, i) for i in range(train_size + validation_size, train_size + validation_size + test_size)]
    for fname in fnames:
        src = os.path.join(images_dir, fname)
        dest = os.path.join(test_pos_dir, fname)
        shutil.copyfile(src, dest)

    fnames = ["{}.{}.png".format(neg, i) for i in range(train_size + validation_size, train_size + validation_size + test_size)]
    for fname in fnames:
        src = os.path.join(images_dir, fname)
        dest = os.path.join(test_neg_dir, fname)
        shutil.copyfile(src, dest)


def serve(**kwargs):
    print("serve")

allowed_cmds = ["run", "capture", "train", "serve"]
commands = {"run": run, "capture": capture, "train": train, "serve": serve}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFix human-robot interface")
    parser.add_argument("--command", dest="command", help="Name of the command to run", required=True)
    parser.add_argument("--positive", dest="positive", help="Prefix of positive class images", required=True)
    parser.add_argument("--negative", dest="negative", help="Prefix of negative class images", required=True)
    parser.add_argument("--size", dest="size", help="Size of images specified in pixels", required=True)
    parser.add_argument("--mode", dest="mode", default="append", help="Determines data capturing mode")
    parser.add_argument("--path", dest="path", default="images", help="Path to folder for captured images")
    parser.add_argument("--model", dest="model", help="Name of model to be trained or run")
    parser.add_argument("--training", dest="training", help="Number of training samples")
    parser.add_argument("--validation", dest="validation", help="Number of validation samples")
    parser.add_argument("--testing", dest="testing", help="Number of test samples")
    args = parser.parse_args()
    
    cmd = args.command
    pos = args.positive
    neg = args.negative
    size = args.size
    path = args.path
    mode = args.mode
    model = args.model
    training = args.training
    validation = args.validation
    testing = args.testing

    if not cmd in allowed_cmds:
        print("{cmd}: command not found. --command must be one of {cmds}".format(cmd=cmd, cmds=allowed_cmds))
        sys.exit(-1)

    if cmd == "train" and not model:
        print("specify a model to be trained")
        sys.exit(-1)
    
    if cmd == "train" and (not training or not validation or not testing):
        print("specify sample sizes for training, validation, and testing")
        sys.exit(-1)

    action = commands[args.command]
    action(path=path, mode=mode, pos=pos, neg=neg, size=size, model=model, training=training, validation=validation, testing=testing)