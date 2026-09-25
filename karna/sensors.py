"""AJO live sensors: mic (arecord) + camera (/dev/video0) into UniBrain, optional."""
import shutil
import subprocess
import numpy as np


def mic_chunk(sec=1.0, sr=8000, live=None):
    import math
    live_flag = {"live": True}
    if shutil.which("arecord") is None:
        live_flag["live"] = False
        return np.random.randn(int(sr * sec)).astype(np.float32) * 0.1, live_flag
    try:
        dur = max(1, int(math.ceil(sec)))
        raw = subprocess.run(["arecord", "-q", "-d", str(dur), "-f", "S16_LE",
                              "-r", str(sr), "-c", "1"], capture_output=True, timeout=dur + 5).stdout
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            live_flag["live"] = False
            return np.zeros(int(sr * sec), dtype=np.float32), live_flag
        return x[:int(sr * sec)], live_flag
    except Exception:
        live_flag["live"] = False
        return np.random.randn(int(sr * sec)).astype(np.float32) * 0.1, live_flag


def cam_frame(w=48, h=48):
    for dev in ("/dev/video0", "/dev/video1"):
        try:
            import os
            if os.path.exists(dev):
                img = _grab_v4l2(dev, w, h)
                if img is not None:
                    return {"live": True, "dev": dev, "frame": img}
                return {"live": True, "dev": dev}
        except Exception:
            pass
    return {"live": False}


def _grab_v4l2(dev, w=48, h=48):
    import shutil
    import subprocess
    if shutil.which("v4l2-ctl") is None and shutil.which("ffmpeg") is None:
        return None
    try:
        if shutil.which("ffmpeg") is not None:
            raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
                                  "-video_size", f"{w}x{h}", "-i", dev, "-frames:v", "1",
                                  "-f", "rawvideo", "-pix_fmt", "gray8", "-"],
                                 capture_output=True, timeout=8).stdout
            a = np.frombuffer(raw, dtype=np.uint8)
            if a.size >= w * h:
                return (a[:w * h].reshape(h, w).astype(np.float32) / 255.0)
    except Exception:
        return None
    return None


if __name__ == "__main__":
    from karna.brain import UniBrain
    b = UniBrain()
    mic, mic_live = mic_chunk(0.2)
    cf = cam_frame()
    print("mic:", mic.shape, "mic_live:", mic_live, "cam:", {k: (v.shape if k == "frame" else v) for k, v in cf.items()})
    r = b.sense(mic, "audio", novelty=0.3)
    print(f"SENSORS_OK audio regions={r['regions']} conf={r['conf']} live={mic_live['live']}")
    if "frame" in cf:
        rv = b.sense(cf["frame"], "video", novelty=0.3)
        print(f"SENSORS_OK video regions={rv['regions']} conf={rv['conf']} live=True")
    else:
        print("SENSORS_OK video live=False (no ffmpeg/v4l2 frame, device present)")
