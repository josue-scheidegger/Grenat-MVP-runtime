#include <stdio.h>
#include <linux/videodev2.h>
int main(){
    printf("VIDIOC_G_FMT=0x%lx\n", (unsigned long)VIDIOC_G_FMT);
    printf("VIDIOC_S_FMT=0x%lx\n", (unsigned long)VIDIOC_S_FMT);
    printf("VIDIOC_REQBUFS=0x%lx\n", (unsigned long)VIDIOC_REQBUFS);
    printf("VIDIOC_QUERYBUF=0x%lx\n", (unsigned long)VIDIOC_QUERYBUF);
    printf("VIDIOC_QBUF=0x%lx\n", (unsigned long)VIDIOC_QBUF);
    printf("VIDIOC_DQBUF=0x%lx\n", (unsigned long)VIDIOC_DQBUF);
    printf("VIDIOC_STREAMON=0x%lx\n", (unsigned long)VIDIOC_STREAMON);
    printf("VIDIOC_STREAMOFF=0x%lx\n", (unsigned long)VIDIOC_STREAMOFF);
    printf("VIDIOC_S_CTRL=0x%lx\n", (unsigned long)VIDIOC_S_CTRL);
    printf("VIDIOC_S_PARM=0x%lx\n", (unsigned long)VIDIOC_S_PARM);
    printf("VIDIOC_S_EXT_CTRLS=0x%lx\n", (unsigned long)VIDIOC_S_EXT_CTRLS);
    printf("sizeof(v4l2_format)=%zu\n", sizeof(struct v4l2_format));
    printf("sizeof(v4l2_requestbuffers)=%zu\n", sizeof(struct v4l2_requestbuffers));
    printf("sizeof(v4l2_buffer)=%zu\n", sizeof(struct v4l2_buffer));
    printf("sizeof(v4l2_control)=%zu\n", sizeof(struct v4l2_control));
    printf("sizeof(v4l2_streamparm)=%zu\n", sizeof(struct v4l2_streamparm));
    printf("sizeof(v4l2_ext_control)=%zu\n", sizeof(struct v4l2_ext_control));
    printf("sizeof(v4l2_ext_controls)=%zu\n", sizeof(struct v4l2_ext_controls));
    return 0;
}
