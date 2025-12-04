import ctypes
import errno
import mmap
import os
import time
from typing import List, Tuple


LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def fourcc(code: str) -> int:
    if len(code) != 4:
        raise ValueError("Pixel format must be exactly 4 characters.")
    code = code.upper()
    return (
        ord(code[0])
        | (ord(code[1]) << 8)
        | (ord(code[2]) << 16)
        | (ord(code[3]) << 24)
    )


IOC_NRBITS = 8
IOC_TYPEBITS = 8
IOC_SIZEBITS = 14
IOC_DIRBITS = 2

IOC_NRSHIFT = 0
IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
IOC_DIRSHIFT = IOC_SIZESHIFT + IOC_SIZEBITS

IOC_NONE = 0
IOC_WRITE = 1
IOC_READ = 2


def _IOC(direction: int, type_: int, number: int, size: int) -> int:
    return (
        (direction << IOC_DIRSHIFT)
        | (type_ << IOC_TYPESHIFT)
        | (number << IOC_NRSHIFT)
        | (size << IOC_SIZESHIFT)
    )


def _IOW(type_: int, number: int, size: int) -> int:
    return _IOC(IOC_WRITE, type_, number, size)


def _IOWR(type_: int, number: int, size: int) -> int:
    return _IOC(IOC_READ | IOC_WRITE, type_, number, size)


class timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class v4l2_format_union(ctypes.Union):
    _fields_ = [
        ("pix", v4l2_pix_format),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class v4l2_format(ctypes.Structure):
    _anonymous_ = ("fmt",)
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
        ("fmt", v4l2_format_union),
    ]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class v4l2_buffer_m(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
    ]


class v4l2_buffer(ctypes.Structure):
    _anonymous_ = ("m",)
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", timeval),
        ("timecode", v4l2_timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", v4l2_buffer_m),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class v4l2_control(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


class v4l2_fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class v4l2_captureparm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", v4l2_fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class v4l2_streamparm_union(ctypes.Union):
    _fields_ = [
        ("capture", v4l2_captureparm),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class v4l2_streamparm(ctypes.Structure):
    _anonymous_ = ("parm",)
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("parm", v4l2_streamparm_union),
    ]


class v4l2_ext_control_value(ctypes.Union):
    _fields_ = [
        ("value", ctypes.c_int32),
        ("value64", ctypes.c_int64),
        ("ptr", ctypes.c_void_p),
        ("p_u32", ctypes.POINTER(ctypes.c_uint32)),
        ("p_u64", ctypes.POINTER(ctypes.c_uint64)),
    ]


class v4l2_ext_control(ctypes.Structure):
    _anonymous_ = ("val",)
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("val", v4l2_ext_control_value),
    ]


class v4l2_ext_controls(ctypes.Structure):
    _fields_ = [
        ("ctrl_class", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("error_idx", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
        ("controls", ctypes.POINTER(v4l2_ext_control)),
    ]


V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_RAW = 0

V4L2_CID_BYPASS_MODE = 0x009A2064

VIDIOC_G_FMT = _IOWR(ord("V"), 4, ctypes.sizeof(v4l2_format))
VIDIOC_S_FMT = _IOWR(ord("V"), 5, ctypes.sizeof(v4l2_format))
VIDIOC_REQBUFS = _IOWR(ord("V"), 8, ctypes.sizeof(v4l2_requestbuffers))
VIDIOC_QUERYBUF = _IOWR(ord("V"), 9, ctypes.sizeof(v4l2_buffer))
VIDIOC_QBUF = _IOWR(ord("V"), 15, ctypes.sizeof(v4l2_buffer))
VIDIOC_DQBUF = _IOWR(ord("V"), 17, ctypes.sizeof(v4l2_buffer))
VIDIOC_STREAMON = _IOW(ord("V"), 18, ctypes.sizeof(ctypes.c_uint32))
VIDIOC_STREAMOFF = _IOW(ord("V"), 19, ctypes.sizeof(ctypes.c_uint32))
VIDIOC_S_CTRL = _IOWR(ord("V"), 28, ctypes.sizeof(v4l2_control))
VIDIOC_S_PARM = _IOWR(ord("V"), 22, ctypes.sizeof(v4l2_streamparm))
VIDIOC_S_EXT_CTRLS = _IOWR(ord("V"), 72, ctypes.sizeof(v4l2_ext_controls))


def xioctl(fd: int, request: int, arg) -> None:
    while True:
        ret = LIBC.ioctl(fd, request, arg)
        if ret != -1:
            return
        err = ctypes.get_errno()
        if err == errno.EINTR:
            continue
        raise OSError(err, os.strerror(err))


def set_control(fd: int, control_id: int, value: int) -> None:
    ctrl = v4l2_control(id=control_id, value=value)
    try:
        xioctl(fd, VIDIOC_S_CTRL, ctypes.byref(ctrl))
        return
    except OSError as exc:
        if exc.errno != errno.ENOTTY:
            raise
    ext_ctrl = v4l2_ext_control(id=control_id, size=0)
    ext_ctrl.value = value
    controls = v4l2_ext_controls()
    controls.ctrl_class = control_id & 0xFFFF0000
    controls.count = 1
    controls.controls = ctypes.pointer(ext_ctrl)
    xioctl(fd, VIDIOC_S_EXT_CTRLS, ctypes.byref(controls))


def configure_format(fd: int, width: int, height: int, pixfmt: int) -> v4l2_pix_format:
    fmt = v4l2_format()
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    fmt.fmt.pix.width = width
    fmt.fmt.pix.height = height
    fmt.fmt.pix.pixelformat = pixfmt
    fmt.fmt.pix.field = V4L2_FIELD_NONE
    fmt.fmt.pix.colorspace = V4L2_COLORSPACE_RAW
    xioctl(fd, VIDIOC_S_FMT, ctypes.byref(fmt))
    return fmt.fmt.pix


def get_format(fd: int) -> v4l2_pix_format:
    fmt = v4l2_format()
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    xioctl(fd, VIDIOC_G_FMT, ctypes.byref(fmt))
    return fmt.fmt.pix


def set_frame_rate(fd: int, fps: int) -> None:
    if fps <= 0:
        return
    parm = v4l2_streamparm()
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    parm.capture.timeperframe.numerator = 1
    parm.capture.timeperframe.denominator = fps
    xioctl(fd, VIDIOC_S_PARM, ctypes.byref(parm))


def request_buffers(fd: int, count: int) -> None:
    req = v4l2_requestbuffers()
    req.count = count
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    req.memory = V4L2_MEMORY_MMAP
    xioctl(fd, VIDIOC_REQBUFS, ctypes.byref(req))
    if req.count < count:
        raise RuntimeError(f"Device provided only {req.count} buffers (requested {count}).")


def release_buffers(fd: int) -> None:
    req = v4l2_requestbuffers()
    req.count = 0
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    req.memory = V4L2_MEMORY_MMAP
    xioctl(fd, VIDIOC_REQBUFS, ctypes.byref(req))


def query_buffer(fd: int, index: int) -> v4l2_buffer:
    buf = v4l2_buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_MMAP
    buf.index = index
    xioctl(fd, VIDIOC_QUERYBUF, ctypes.byref(buf))
    return buf


def map_buffers(fd: int, count: int) -> List[Tuple[mmap.mmap, int]]:
    buffers: List[Tuple[mmap.mmap, int]] = []
    for index in range(count):
        buf = query_buffer(fd, index)
        mm = mmap.mmap(
            fd,
            buf.length,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
            offset=buf.m.offset,
        )
        buffers.append((mm, buf.length))
    return buffers


def queue_buffer(fd: int, index: int, length: int) -> None:
    buf = v4l2_buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_MMAP
    buf.index = index
    buf.length = length
    buf.bytesused = length
    xioctl(fd, VIDIOC_QBUF, ctypes.byref(buf))


def dequeue_buffer(fd: int, timeout: float) -> v4l2_buffer:
    end_time = None if timeout is None else (time.monotonic() + timeout)
    while True:
        buf = v4l2_buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        try:
            xioctl(fd, VIDIOC_DQBUF, ctypes.byref(buf))
            return buf
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EINTR):
                if end_time is not None and time.monotonic() >= end_time:
                    raise TimeoutError("Timed out waiting for frame.") from None
                time.sleep(0.005)
                continue
            raise


def start_stream(fd: int) -> None:
    buf_type = ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE)
    xioctl(fd, VIDIOC_STREAMON, ctypes.byref(buf_type))


def stop_stream(fd: int) -> None:
    buf_type = ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE)
    xioctl(fd, VIDIOC_STREAMOFF, ctypes.byref(buf_type))


class V4L2Capture:
    """Minimal V4L2 capture helper using ioctl + mmap."""

    def __init__(self, device: str, non_blocking: bool = True):
        self.device = device
        flags = os.O_RDWR | (os.O_NONBLOCK if non_blocking else 0)
        self.fd = os.open(device, flags)
        self.buffers: List[Tuple[mmap.mmap, int]] = []
        self.streaming = False
        self.format: v4l2_pix_format | None = None

    def configure(
        self,
        width: int,
        height: int,
        pixfmt: int,
        bypass_mode: int | None = None,
        fps: int | None = None,
    ) -> v4l2_pix_format:
        if bypass_mode is not None:
            set_control(self.fd, V4L2_CID_BYPASS_MODE, bypass_mode)
        fmt = configure_format(self.fd, width, height, pixfmt)
        if fps:
            set_frame_rate(self.fd, fps)
        self.format = fmt
        return fmt

    def current_format(self) -> v4l2_pix_format:
        self.format = get_format(self.fd)
        return self.format

    def start(self, buffer_count: int = 4) -> None:
        if self.streaming:
            return
        request_buffers(self.fd, buffer_count)
        self.buffers = map_buffers(self.fd, buffer_count)
        for index, (_, length) in enumerate(self.buffers):
            queue_buffer(self.fd, index, length)
        start_stream(self.fd)
        self.streaming = True

    def read(self, timeout: float) -> bytes:
        if not self.streaming:
            raise RuntimeError("Call start() before read().")
        buf = dequeue_buffer(self.fd, timeout)
        mm, length = self.buffers[buf.index]
        data = bytes(mm[: buf.bytesused])
        queue_buffer(self.fd, buf.index, length)
        return data

    def stop(self) -> None:
        if not self.streaming:
            return
        try:
            stop_stream(self.fd)
        finally:
            for mm, _ in self.buffers:
                mm.close()
            self.buffers = []
            try:
                release_buffers(self.fd)
            except OSError:
                pass
            self.streaming = False

    def close(self) -> None:
        try:
            self.stop()
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def __enter__(self) -> "V4L2Capture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
