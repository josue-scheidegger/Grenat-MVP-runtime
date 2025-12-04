#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CHECK(x) do { if ((x) == -1) { perror(#x); exit(1); } } while (0)

static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

int main(void) {
    const char *dev = "/dev/video0";
    int fd = open(dev, O_RDWR);
    CHECK(fd);

    struct v4l2_control ctrl = {
        .id = 0x009a2064,
        .value = 0,
    };
    CHECK(xioctl(fd, VIDIOC_S_CTRL, &ctrl));

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 2048;
    fmt.fmt.pix.height = 1536;
    fmt.fmt.pix.pixelformat = v4l2_fourcc('R','G','1','0');
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    CHECK(xioctl(fd, VIDIOC_S_FMT, &fmt));

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    CHECK(xioctl(fd, VIDIOC_REQBUFS, &req));

    struct {
        void *start;
        size_t length;
    } buffers[4];

    for (int i = 0; i < 4; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        CHECK(xioctl(fd, VIDIOC_QUERYBUF, &buf));
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            perror("mmap");
            return 1;
        }
    }

    for (int i = 0; i < 4; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        CHECK(xioctl(fd, VIDIOC_QBUF, &buf));
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    CHECK(xioctl(fd, VIDIOC_STREAMON, &type));

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    CHECK(xioctl(fd, VIDIOC_DQBUF, &buf));

    FILE *out = fopen("frame_c.bin", "wb");
    fwrite(buffers[buf.index].start, 1, buf.bytesused, out);
    fclose(out);

    CHECK(xioctl(fd, VIDIOC_STREAMOFF, &type));

    for (int i = 0; i < 4; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }

    close(fd);
    return 0;
}
