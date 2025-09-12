import json
import numpy as np
from skel_video import make_skel_video
from stack_videos import stack_videos
from tests_resources.skel_structure import SKELETON_STRUCTURE
from plot_mel_spectrogram import make_mel_spec_image
import matplotlib
matplotlib.use('Agg')  # to avoid Qt error when testing function that use matplotlib.pyplot


def skel_video_test():
    with open("./tests_resources/0000.json", "r") as f:
        data_skel = json.load(f)
    with open("./tests_resources/0000.txt", "r", encoding="utf-8") as f:
        data_txt = f.readlines()
    text = "".join(data_txt)
    text = text.replace("\n", "")
    skel_sequence = np.array([np.array(skel).reshape(-1, 3) for skel in data_skel])
    # my_long_text = "This is a very long sentence to test how it will split the sentence in the video with the tool! Here is another one so that it is long."
    make_skel_video(
        skel_sequence=skel_sequence,
        structure=SKELETON_STRUCTURE,
        output_folder="./tests_outputs",
        output_name="0000_skel",
        attach_image="./tests_resources/img.png",
        # attach_text=my_long_text
        attach_text=text,
    )
    print("==> Created skeletal poses video!")


def stack_video_test():
    my_vid1, label1 = "./tests_resources/0000.mp4", "This is a Raw Video"
    my_vid2, label2= "./tests_outputs/0000_skel.mp4", "This is a Skel Sequence"
    stack_videos(
        vid1=my_vid1, vid2=my_vid2,
        output_folder="./tests_outputs",
        output_name="0000_stack",
        width=1000, height=800, frame_rate=30,
        txt1=label1, txt2=label2,
    )
    print("==> Created a new video by stacking videos!")


def mel_spec_plot_test():
    mel_spec = 3 * (1 - 2 * np.random.random((100, 80)))
    make_mel_spec_image(mel_spec=mel_spec, save_fpath="./tests_outputs/test_mel_spec.png")
    print("==> Created a Mel Spectrogram image!")


if __name__ == "__main__":
    skel_video_test()
    stack_video_test()
    mel_spec_plot_test()