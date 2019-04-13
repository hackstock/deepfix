import argparse
import sys
import os
import shutil
from core.faces.loader import Recognizer

def run(**kwargs):
    print("deepfix is running")

def capture(**kwargs):
    path = kwargs["path"]
    mode = kwargs["mode"]

    if mode == "reset" and os.path.exists(path):
        shutil.rmtree(path)
    elif mode == "append" and not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, "positives"))
        os.mkdir(os.path.join(path, "negatives"))

    rec = Recognizer(src=0, haar_path="./haarcascade_frontalface.xml")
    rec.run()

def train(**kwargs):
    print("training")

def serve(**kwargs):
    print("serve")

allowed_cmds = ["run", "capture", "train", "serve"]
commands = {"run": run, "capture": capture, "train": train, "serve": serve}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFix human-robot interface")
    parser.add_argument("--command", dest="command", help="Name of the command to run", required=True)
    parser.add_argument("--mode", dest="mode", default="append", help="Determines data capturing mode")
    parser.add_argument("--path", dest="path", default="images", help="Path to folder for captured images")
    args = parser.parse_args()
    
    cmd = args.command
    path = args.path
    mode = args.mode

    if not cmd in allowed_cmds:
        print("{cmd}: command not found. --command must be one of {cmds}".format(cmd=cmd, cmds=allowed_cmds))
        sys.exit(-1)

    action = commands[args.command]
    action(path=path, mode=mode)