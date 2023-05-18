import av
import av.error
import av.video
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoAV:

    def __init__(self, video):
        self.video = video

        container = av.open(video)
        stream = container.streams.video[0]
        self.width = stream.codec_context.width
        self.height = stream.codec_context.height
        self._frame_count = stream.frames

        def _get_frame_rate(stream: av.video.stream.VideoStream):
            if stream.average_rate.denominator and stream.average_rate.numerator:
                return float(stream.average_rate)
            if stream.time_base.denominator and stream.time_base.numerator:
                return 1.0 / float(stream.time_base)
            else:
                raise ValueError("Unable to determine FPS")

        self.frame_rate = _get_frame_rate(stream)
        container.close()

    def frames(self, **reformat_kwargs):
        container = av.open(self.video)
        for index, frame in enumerate(container.decode(video=0)):
            yield index, frame.to_ndarray(**reformat_kwargs)
        container.close()

    @property
    def frame_count(self):
        if self._frame_count == 0:
            index = 0
            container = av.open(self.video)
            try:
                for index, _ in enumerate(container.decode(video=0)):
                    continue
            except av.error.EOFError:
                pass
            container.close()
            self._frame_count = index + 1
        return self._frame_count


class VideoPredict(Dataset):
    r"""
    A predict dataset for a pair of videos

    args:
        video_ref (str): the path to the reference video
        video_dis (str): the path to the distorted video
    """

    def __init__(self, video_ref, video_dis):

        self.video_ref = video_ref
        self.video_dis = video_dis

        self.ref = VideoAV(video_ref)
        self.dis = VideoAV(video_dis)
        # check frame count and frame rate
        assert self.ref.frame_count == self.dis.frame_count and self.ref.frame_rate == self.dis.frame_rate

        self.frame_count = self.ref.frame_count
        self.frame_rate = self.ref.frame_rate
        if self.frame_rate <= 30:
            self.stride_t = 2
        elif self.frame_rate <= 60:
            self.stride_t = 4
        else:
            raise ValueError('Unsupported fps')

        if max(self.ref.height, self.dis.height) >= 1080:
            self.frame_width = 1920
            self.frame_height = 1080
        else:
            self.frame_width = 1280
            self.frame_height = 720
        print(f'Calculated resolution is {self.frame_width}x{self.frame_height}.')

        self.ref_frames = self.ref.frames(width=self.frame_width, height=self.frame_height, format='rgb24')
        self.dis_frames = self.dis.frames(width=self.frame_width, height=self.frame_height, format='rgb24')

    def __getitem__(self, index):
        stride_t = self.stride_t
        while stride_t > 1:
            _ = next(self.ref_frames)
            _ = next(self.dis_frames)
            stride_t -= 1

        _, ref = next(self.ref_frames)
        ref = (ref - 0.5) / 0.5
        ref = np.transpose(ref, (2, 0, 1))
        ref = torch.from_numpy(ref).type(torch.FloatTensor)

        _, dis = next(self.dis_frames)
        dis = (dis - 0.5) / 0.5
        dis = np.transpose(dis, (2, 0, 1))
        dis = torch.from_numpy(dis).type(torch.FloatTensor)

        return ref, dis

    def __len__(self):
        return self.frame_count
