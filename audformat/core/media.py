import typing

from audformat.core import define
from audformat.core.common import HeaderBase


class Media(HeaderBase):
    r"""Media information.

    File ``format`` is always converted to lower case.

    Args:
        type: media type
        format: file format (e.g. 'wav', 'flac', 'mp4')
        sampling_rate: audio sampling rate in Hz
        channels: number of audio channels
        bit_depth: audio bit depth
        video_fps: video rate in frames per seconds
        video_resolution: video resolution in pixels (e.g. ``[800, 600]``)
        video_channels: number of channels per pixel (e.g. 3 for RGB)
        video_depth: number of bits per video channel
        description: media description
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``type`` is passed

    Example:
        >>> Media(
        ...     type=define.MediaType.AUDIO,
        ...     format='wav',
        ...     sampling_rate=16000,
        ...     channels=2,
        ... )
        {type: audio, format: wav, channels: 2, sampling_rate: 16000}

    """
    def __init__(
            self,
            type: define.MediaType = define.MediaType.OTHER,
            *,
            format: str = None,
            sampling_rate: int = None,
            channels: int = None,
            bit_depth: int = None,
            video_fps: int = None,
            video_resolution: typing.Sequence[int] = None,
            video_channels: int = None,
            video_depth: int = None,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)
        define.MediaType.assert_has_attribute_value(type)

        self.type = type
        r"""Media type"""
        self.format = None if format is None else str(format).lower()
        r"""File format"""

        self.channels = channels
        r"""Audio channels"""
        self.bit_depth = bit_depth
        r"""Audio bit depth"""
        self.sampling_rate = sampling_rate
        r"""Audio sampling rate in Hz"""

        self.video_fps = video_fps
        r"""Video frames per second"""
        self.video_resolution = None if video_resolution is None \
            else list(video_resolution)
        r"""Video resolution"""
        self.video_channels = video_channels
        r"""Video channels per pixel"""
        self.video_depth = video_depth
        r"""Video bit depth"""
