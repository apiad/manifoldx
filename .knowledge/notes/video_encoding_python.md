# Video Encoding in Python: Research Summary

This document summarizes research on video encoding capabilities in Python, focusing on OpenCV video writing, format/codec options, and alternatives.

## 1. OpenCV (cv2) Video Writing Capabilities

OpenCV provides the `cv2.VideoWriter` class for writing video files. Here's how to use it:

### Basic Usage

```python
import cv2
import numpy as np

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'mp4v' for MP4
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

# Write frames
for frame in frames:
    out.write(frame)

# Release when done
out.release()
```

### Key Parameters

- **FourCC code**: 4-byte code specifying the video codec
- **FPS**: Frames per second (e.g., 30.0, 60.0)
- **Frame size**: Tuple of (width, height)
- **isColor**: Boolean flag for color/grayscale

## 2. Format and Codec Options for MP4

### Recommended Codecs for MP4

| Codec FourCC | Description | Platform Notes |
|--------------|-------------|----------------|
| `'mp4v'` | MPEG-4 Part 2 | Widely supported, good compatibility |
| `'avc1'` / `'H264'` | H.264/AVC | Requires OpenCV with FFMPEG backend |
| `'X264'` | x264 encoder | Better quality, requires ffmpeg |
| `'XVID'` | Xvid encoder | Good compatibility, larger files |
| `'hvc1'` | HEVC/H.265 | For newer devices |

### For H.264 (Best Quality/Size Ratio)

```python
# Method 1: Using H264 codec
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))

# Method 2: Using avc1
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))
```

### Important Notes

- OpenCV's support for specific codecs depends on the backend (FFmpeg, GStreamer, etc.)
- On Windows: DIVX, mp4v work well
- On Linux: XVID, H264, mjpeg work well
- On macOS: mjpeg, mp4v work well

## 3. Frame Rate and Resolution Handling

### Setting Resolution

```python
# Get current properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set specific resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

### Setting Frame Rate

```python
# Set FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Or use set on VideoWriter
out.set(cv2.CAP_PROP_FPS, 60)
```

### Properties Available

| Property | Description |
|----------|-------------|
| `CAP_PROP_FRAME_WIDTH` | Frame width |
| `CAP_PROP_FRAME_HEIGHT` | Frame height |
| `CAP_PROP_FPS` | Frames per second |
| `CAP_PROP_FOURCC` | FourCC code |
| `CAP_PROP_FRAME_COUNT` | Total frame count |
| `CAP_PROP_BRIGHTNESS` | Brightness |
| `CAP_PROP_CONTRAST` | Contrast |

### Resolution Alignment

For best results with H.264, align frame dimensions to macroblock size (typically 16):

```python
def align_resolution(width, height, block_size=16):
    """Align dimensions to block size"""
    return (width // block_size * block_size, 
            height // block_size * block_size)
```

## 4. Alternatives to OpenCV

### A. imageio-ffmpeg

A lightweight FFmpeg wrapper that includes binary executables.

**Installation:**
```bash
pip install imageio-ffmpeg
```

**Basic Usage:**
```python
import imageio_ffmpeg
import numpy as np

# Get path to ffmpeg executable
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# Write video
writer = imageio_ffmpeg.write_frames(
    'output.mp4',
    size=(1920, 1080),
    fps=30,
    codec='libx264',
    quality=5  # 0-10, lower is better quality
)
writer.send(None)  # seed the generator

for frame in frames:
    writer.send(frame)

writer.close()
```

**Key Features:**
- Includes FFmpeg binaries for all platforms
- Simple generator-based API
- Supports quality settings
- Can add audio

### B. PyAV (Recommended for Advanced Use)

A more feature-complete binding for FFmpeg libraries.

**Installation:**
```bash
pip install av
```

**Basic Usage:**
```python
import av
import numpy as np

# Create output container
output = av.open('output.mp4', 'w')

# Add video stream
stream = output.add_stream('h264', rate=30)
stream.width = 1920
stream.height = 1080
stream.options = {'crf': '23', 'preset': 'medium'}

# Write frames
for frame_data in frames:
    frame = av.VideoFrame.from_numpy(frame_data)
    for packet in stream.encode(frame):
        output.mux(packet)

# Flush encoder
for packet in stream.encode():
    output.mux(packet)

output.close()
```

**Key Features:**
- Direct access to FFmpeg libraries
- Fine-grained control over encoding
- Supports audio encoding
- Better performance for large files

### C. ffmpeg-python

A Python wrapper for FFmpeg command-line tool.

**Installation:**
```bash
pip install ffmpeg-python
```

**Basic Usage:**
```python
import ffmpeg
import numpy as np

# Process video
stream = ffmpeg.input('input.mp4')
stream = ffmpeg.filter(stream, 'scale', 1280, 720)
stream = ffmpeg.output(stream, 'output.mp4', vcodec='libx264', crf=23)
ffmpeg.run(stream)
```

**Key Features:**
- Chainable filters
- Good for transcoding existing videos
- Not ideal for real-time frame-by-frame encoding

## 5. Graphics Libraries (pygfx) to Video

### Pygfx Overview

Pygfx is a modern graphics library built on wgpu. While it doesn't directly output video files, you can capture rendered frames and encode them.

**Installation:**
```bash
pip install pygfx glfw
```

**Capturing Frames from Pygfx:**

```python
import pygfx as gfx
import numpy as np
from PIL import Image

# Create scene and camera (standard pygfx setup)
# ... setup code ...

# To capture a frame, render to a texture
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Get the render result as a texture
result = renderer.render(scene, camera)
texture = result["color"]

# Convert to numpy array
frame = texture.data  # This is a numpy array

# Now use OpenCV or imageio-ffmpeg to save
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
out.write(frame)
out.release()
```

### Complete Workflow Example

```python
import pygfx as gfx
import numpy as np
import imageio_ffmpeg

def render_and_save_video(scene, camera, renderer, output_path, fps=30):
    """Render animation and save as video"""
    
    # Prepare writer
    writer = imageio_ffmpeg.write_frames(
        output_path,
        size=renderer.get_size(),
        fps=fps,
        codec='libx264',
        quality=5
    )
    writer.send(None)
    
    # Animation loop
    num_frames = 300  # for example
    for i in range(num_frames):
        # Update your scene (rotation, etc.)
        update_scene(scene, i)
        
        # Render
        result = renderer.render(scene, camera)
        
        # Get frame data
        frame = result["color"].data
        
        # Convert RGBA to RGB if needed
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        
        # Send to writer
        writer.send(frame)
    
    writer.close()
```

## 6. Best Approaches Summary

### For Simple Use Cases

- **OpenCV (`cv2.VideoWriter`)**: Best for quick prototyping and simple video writing tasks. Limited codec support but easy to use.

### For Production/Quality Video

- **PyAV**: Best for full control over encoding parameters, audio/video synchronization, and professional quality output.

- **imageio-ffmpeg**: Good balance between simplicity and quality. Includes binaries, making distribution easy.

### For Graphics/Scientific Visualization

- **Pygfx + imageio-ffmpeg**: Render with Pygfx, capture frames, encode with imageio-ffmpeg for best quality.

### Recommendation

For most use cases involving ManifoldX visualization:

1. **Use imageio-ffmpeg** for video encoding (simple, included binaries, good quality)
2. **Use OpenCV** only if you already have it in dependencies and need basic functionality
3. **Use PyAV** if you need advanced features like audio, precise control over encoding parameters

## 7. Quick Reference Code

### Recommended Setup for Video Encoding

```python
import numpy as np
import imageio_ffmpeg

def create_video_writer(path, width, height, fps=30, codec='libx264'):
    """Create a video writer with recommended settings"""
    writer = imageio_ffmpeg.write_frames(
        path,
        size=(width, height),
        fps=fps,
        codec=codec,
        quality=5,  # 0-10, lower = better quality
        pix_fmt_in='rgb24'
    )
    writer.send(None)  # Seed the generator
    return writer

def save_video(path, frames, width, height, fps=30):
    """Save numpy array frames to video"""
    writer = create_video_writer(path, width, height, fps)
    
    for frame in frames:
        # Ensure frame is correct format
        if isinstance(frame, np.ndarray):
            writer.send(frame.tobytes())
    
    writer.close()
```

### Example with Pygfx Integration

```python
import pygfx as gfx
import imageio_ffmpeg
import numpy as np

def render_animation_to_video(scene, camera, renderer, output_path, fps=30):
    """Render animation and save to video file"""
    
    # Get render size
    width, height = renderer.get_size()
    
    # Create writer
    writer = imageio_ffmpeg.write_frames(
        output_path,
        size=(width, height),
        fps=fps,
        codec='libx264',
        quality=4
    )
    writer.send(None)
    
    # Render frames
    for _ in range(300):  # Number of frames
        # Update your animation here
        # ...
        
        # Render the scene
        result = renderer.render(scene, camera)
        
        # Extract frame data
        frame = result["color"].data
        
        # Convert RGBA to RGB if needed
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        
        # Write frame
        writer.send(frame.tobytes())
    
    writer.close()
```

## 8. Additional Resources

- OpenCV VideoWriter: https://docs.opencv.org/
- imageio-ffmpeg: https://github.com/imageio/imageio-ffmpeg
- PyAV: https://github.com/PyAV-Org/PyAV
- Pygfx: https://github.com/pygfx/pygfx
- FFmpeg codecs: https://ffmpeg.org/ffmpeg-codecs.html