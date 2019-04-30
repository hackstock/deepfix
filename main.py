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

    if mode == "reset" and os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    elif mode == "append" and not os.path.exists(path):
        os.mkdir(path)

    rec = Recognizer(src=0, haar_path="./haarcascade_frontalface.xml", pos_prefix=pos, neg_prefix=neg, size=size)
    rec.run()

def train(**kwargs):
    known_models = ["convnet", "logisticreg", "xception"]
    model = kwargs["model"]
    if not model in known_models:
        print("{model}: model not found. --model must be one of {models}".format(model=model, models=known_models))
        sys.exit(-1)

    network = None
    if model == "convnet":
        network = ConvNet()
        print(network.model.summary())

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
    args = parser.parse_args()
    
    cmd = args.command
    pos = args.positive
    neg = args.negative
    size = args.size
    path = args.path
    mode = args.mode
    model = args.model

    if not cmd in allowed_cmds:
        print("{cmd}: command not found. --command must be one of {cmds}".format(cmd=cmd, cmds=allowed_cmds))
        sys.exit(-1)

    if cmd == "train" and not model:
        print("specify a model to be trained")
        sys.exit(-1)

    action = commands[args.command]
    action(path=path, mode=mode, pos=pos, neg=neg, size=size, model=model)