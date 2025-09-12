import os
import subprocess


def stack_videos(
        vid1: str, vid2: str,
        output_folder: str,
        output_name: str,
        width: int = None, height: int = None, frame_rate: float = None,
        txt1: str = "", txt2: str = "",
):
    w, h, fr = 800, 800, 30
    adjust_videos = False
    if width is not None:
        w = width
        adjust_videos = True
    if height is not None:
        h = height
        adjust_videos = True
    if frame_rate is not None:
        fr = frame_rate
        adjust_videos = True

    videos_paths = [os.path.abspath(vid) for vid in [vid1, vid2]]

    # 1) Resizing videos + adjusting frame rate (creating temp files)
    temp_videos_paths = None
    if adjust_videos:
        temp_videos_paths = []
        for i, vid in enumerate(videos_paths):
            temp_vid = os.path.abspath(f"{output_folder}/temp{i}.mp4")
            cmd_adjust = [
                "ffmpeg",
                "-i", os.path.abspath(vid),  # Input video
                "-vf", f"scale={w}:{h}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",  # Encoding options
                "-r", f"{fr}",  # Normalize frame rate
                # "-an",  # Remove audio (optional)
                "-y", temp_vid  # Overwrite output file if it exists
            ]
            subprocess.run(cmd_adjust, check=True)
            temp_videos_paths += [temp_vid]
        videos_paths = temp_videos_paths

    # 2) Creating side-by-side videos adding a text label for each of them (stacking videos)
    output_video_fpath = f"{output_folder}/{output_name}.mp4"
    cmd_stack = [
        "ffmpeg",
        "-i", videos_paths[0], "-i", videos_paths[1],  # Input files
        "-filter_complex",
        f"""
        [0:v]pad={w}:{h + 50}:0:50,drawtext=text='{txt1}':x=(w-text_w)/2:y=10:fontsize=24:fontcolor=white[v0];
        [1:v]pad={w}:{h + 50}:0:50,drawtext=text='{txt2}':x=(w-text_w)/2:y=10:fontsize=24:fontcolor=white[v1];
        [v0][v1]hstack=inputs=2[v]      
        """,  # [0:a] aformat=sample_fmts=fltp:sample_rates=16000:channel_layouts=stereo[a]
        "-map", "[v]",  # Video output
        # "-map", "[a]",  # Audio output
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",  # Encoding for video
        # "-c:a", "aac", "-b:a", "160k",  # Encoding for audio
        "-y", os.path.abspath(output_video_fpath)  # Overwrite output file if exists
    ]
    subprocess.run(cmd_stack, check=True)

    # 3) Removing temporary files if any
    if temp_videos_paths:
        for f in temp_videos_paths:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed temporary file: {f}")
