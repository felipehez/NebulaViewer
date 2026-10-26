"""Video output for NebulaViewer: H.264 MP4 straight from the GPU frames (ffmpeg bundled with imageio-ffmpeg),
optionally also a PNG sequence (for compositing / re-editing).

The recorded image is a crop of the viewport camera (the guide rectangle drawn in the window), rendered
again at the video resolution with its own Renderer, so the window can stay small while the video is 4K.
"""
import datetime, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, width, height); "window" = viewport size
VIDEO_SIZES = [("1920x1080", 1920, 1080), ("2560x1440", 2560, 1440), ("3840x2160", 3840, 2160),
               ("1080x1080", 1080, 1080), ("1080x1350 (4:5)", 1080, 1350), ("1080x1920 (9:16)", 1080, 1920),
               ("1280x720", 1280, 720), ("window", 0, 0)]


def crop_tans(cam_tans, vw, vh, w, h):
    """camera tangents of the largest w:h rectangle centred in a vw x vh viewport (+ that rectangle in px)"""
    tx, ty = cam_tans
    a_v, a = vw / vh, w / h
    if a >= a_v:                 # wider than the viewport: full width, letterboxed
        gw, gh = vw, vw / a
    else:
        gw, gh = vh * a, vh
    return (tx * gw / vw, ty * gh / vh), ((vw - gw) * 0.5, (vh - gh) * 0.5, gw, gh)


class Recorder:
    """rec = Recorder(); rec.start(...); rec.add(rgba_uint8_HxWx4); ...; rec.stop() -> mp4 path"""

    def __init__(self):
        self.gen = None
        self.path = None
        self.n = 0
        self.t0 = 0.0
        self.png_dir = None
        self.size = None

    @property
    def active(self):
        return self.gen is not None

    def start(self, name, size, fps=30, crf=16, png=False, out_dir=None):
        import imageio_ffmpeg
        w, h = size
        w, h = w - (w % 2), h - (h % 2)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir or os.path.join(HERE, "videos")
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{name}_{stamp}.mp4")
        self.size = (w, h)
        self.fps = fps
        self.gen = imageio_ffmpeg.write_frames(
            self.path, (w, h), pix_fmt_in="rgb24", pix_fmt_out="yuv420p", fps=fps, codec="libx264",
            quality=None, macro_block_size=2, ffmpeg_log_level="error",
            output_params=["-crf", str(int(crf)), "-preset", "slow", "-tune", "film", "-movflags", "+faststart"])
        self.gen.send(None)
        self.png_dir = None
        if png:
            self.png_dir = os.path.splitext(self.path)[0] + "_png"
            os.makedirs(self.png_dir, exist_ok=True)
        self.n = 0
        self.t0 = time.perf_counter()
        return self.path

    def add(self, rgba):
        """rgba: (H, W, 4) uint8, top row first"""
        w, h = self.size
        img = np.ascontiguousarray(rgba[:h, :w, :3])
        self.gen.send(img.tobytes())
        if self.png_dir:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(self.png_dir, f"f{self.n:05d}.png"), compress_level=1)
        self.n += 1

    def stop(self):
        if self.gen is None:
            return None
        self.gen.close()
        self.gen = None
        return self.path

    def status(self):
        if not self.active:
            return ""
        return f"REC {self.n} frames ({self.n / self.fps:.1f} s of video) {self.size[0]}x{self.size[1]}"
