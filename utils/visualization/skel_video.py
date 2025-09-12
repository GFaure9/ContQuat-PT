import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import cm
from typing import Tuple


def generate_bgr_colors(num_colors: int, colormap: str = 'viridis'):
    cmap = cm.get_cmap(colormap, num_colors)
    return [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in cmap.colors]


def write_skel_img(
        img: cv2.typing.MatLike,
        pts: np.ndarray,  # with shape (N_points, 3) or (N_points, 2)
        structure: Tuple,
        translate: np.ndarray = None,
        scale: float = 1/5,
        fancy_colors: bool = True,
):
    pts = pts[:, :2] * img.shape[0]

    translate = translate if translate is not None else np.zeros(2)
    between_shoulders_id = 1

    # Resizing and centering skeleton
    # --------------------------------------------------------
    # i/ scaling + re-centering
    pts *= scale

    img_ct = np.array([img.shape[0], img.shape[1]]) / 2
    c = pts[between_shoulders_id]
    pts += -(c - img_ct)

    # ii/ translation
    pts += translate
    # --------------------------------------------------------

    # Drawing lines and nodes
    # --------------------------------------------------------
    # utils
    if fancy_colors:
        colors = generate_bgr_colors(len(structure), colormap="turbo")  # "viridis"  "magma"
    to_int_tuple = lambda point: (int(point[0]), int(point[1]))

    # write connections (lines)
    for i, bone in enumerate(structure):
        id1, id2 = bone[0], bone[1]
        x1, x2 = to_int_tuple(pts[id1]), to_int_tuple(pts[id2])
        if x1[0] >= 0 and x2[0] >= 0:
            color = colors[i] if fancy_colors else [0, 0, 0]
            cv2.line(img, x1, x2, color=color, thickness=4)

    # write nodes (circles)
    for i, x in enumerate(pts):
        x = to_int_tuple(x)
        if x[0] >= 0:
            color = [0, 0, 0] if fancy_colors else [0, 0, 225]
            cv2.circle(img, x, radius=1, color=color, thickness=4)


def make_skel_video(
        skel_sequence: np.ndarray,
        output_folder: str,
        output_name: str,
        structure: Tuple,
        translate: np.ndarray = np.array([0, -100]),
        scale: float = 1/5,
        fps: float = 30,
        attach_image: str = None,
        attach_text: str = None,
):
    # N.B: `skel_sequence` is of the form:
    #  [skel_pts1, skel_pts2, ..., skel_ptsT] with
    #   skel_pts{i} = np.array([ [x0, y0, z0],
    #                               ...
    #                            [xn, yn, zn] ])

    output_video_path = f"{output_folder}/{output_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 800, 800

    attach_img = None
    if attach_image:
        attach_img = cv2.imread(attach_image)
        if attach_img is not None:
            attach_img = cv2.resize(attach_img, (w, h))  # match height with skeleton frame

    final_w = w * 2 if attach_img is not None else w
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (final_w, h))

    for skel_arr in tqdm(skel_sequence):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 255
        write_skel_img(img=frame, pts=skel_arr, structure=structure, translate=translate, scale=scale)

        if attach_img is not None:
            combined_frame = np.hstack((frame, attach_img))  # stack images side by side
        else:
            combined_frame = frame

        if attach_text:
            # preprocessing the text for correct display with `cv2.putText`
            # ----- to handle 'strange' quotes
            attach_text = attach_text.replace('“', '"').replace('”', '"')
            # ----- to handle non-ASCII characters (here German ones)
            attach_text = attach_text.replace("ä", "ae")
            attach_text = attach_text.replace("ö", "oe")
            attach_text = attach_text.replace("ü", "ue")
            attach_text = attach_text.replace("ß", "ss")

            list_words = attach_text.split(" ")
            k_words = 10
            for i in range(len(list_words) // k_words + 1):
                sublist_words = list_words[i * k_words: (i + 1) * k_words]
                subtext = " ".join(sublist_words)
                cv2.putText(
                    combined_frame,
                    subtext,
                    (50, 30 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1, cv2.LINE_AA
                )

        video_writer.write(combined_frame)

    video_writer.release()
