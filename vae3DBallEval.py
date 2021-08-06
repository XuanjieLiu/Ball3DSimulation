from vae3DBall import VAE, load_tensor, MODEL_PATH, LATENT_CODE_NUM, LAST_CN_NUM, LAST_H, LAST_W, IMG_CHANNEL
from tkinter import *
from PIL import Image, ImageTk
from torchvision.utils import save_image
import os
import torch

RANGE = 3.2
CODE_LEN = LATENT_CODE_NUM
IMG_PATH = 'vae3DBallEval_ImgBuffer'
IGM_NAME = IMG_PATH + "/test.png"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_img_path():
    if not os.path.isdir(IMG_PATH):
        os.mkdir(IMG_PATH)


def init_codes():
    codes = []
    for i in range(0, CODE_LEN):
        codes.append(0.0)
    return codes


def init_vae():
    vae = VAE().cuda()
    if os.path.exists(MODEL_PATH):
        vae.load_state_dict(load_tensor(MODEL_PATH))
        print(f"Model is loaded")
    return vae


class TestUI:
    def __init__(self):
        init_img_path()
        self.win = Tk()
        self.code = init_codes()
        self.scale_list = self.init_scale_list()
        self.vae = init_vae()
        self.photo = None
        self.label = Label(self.win)
        self.label.pack(side=RIGHT)

    def on_scale_move(self, value, index):
        self.code[index] = float(value)
        self.scale_list[index].set(float(value))
        self.recon_img()
        self.load_img()
        print(self.code)

    def recon_img(self):
        codes = torch.tensor(self.code, dtype=torch.float).to(device)
        sample = self.vae.decoder(self.vae.fc2(codes).view(1, LAST_CN_NUM, LAST_H, LAST_W))
        save_image(sample.data.view(1, IMG_CHANNEL, 80, 120), IGM_NAME)

    def load_img(self):
        self.photo = ImageTk.PhotoImage(Image.open(IGM_NAME))
        self.label.config(image=self.photo)

    def init_scale_list(self):
        scale_list = []
        for i in range(0, CODE_LEN):
            scale = Scale(
                self.win,
                variable=DoubleVar(value=self.code[i]),
                command=lambda value, index=i: self.on_scale_move(value, index),
                from_=0 - RANGE / 2,
                to=0 + RANGE / 2,
                resolution=0.1,
                length=600,
                tickinterval=0.2
            )
            scale.pack(side=LEFT)
            scale_list.append(scale)
        return scale_list


if __name__ == "__main__":
    test_ui = TestUI()
    test_ui.win.mainloop()
