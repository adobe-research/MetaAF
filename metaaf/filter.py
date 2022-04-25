from abc import ABCMeta, abstractmethod

import haiku as hk
from haiku.initializers import RandomNormal
import jax.numpy as jnp
from jax import jit
import jax

from metaaf.complex_utils import complex_zeros


class BufferedFilterMixin(metaclass=ABCMeta):
    def __init__(
        self,
        window_size,
        hop_size,
        pad_size,
        n_frames,
        n_in_chan,
        n_out_chan,
        is_real=False,
        name="Filter",
    ):
        """This is the buffered filter base-class that manages all buffering and online processing. It is overridden by the OLA and OLA classes.

        Args:
            window_size (_type_): Int samples in a window
            hop_size (_type_): Int hop size in samples
            pad_size (_type_): Int samples to pad window by
            n_frames (_type_): Int number of delayed frames to track
            n_in_chan (_type_): Int number of input channels
            n_out_chan (_type_): Int number of output channels
            is_real (bool, optional): If the input signal is real and shold use rfft/irfft. Defaults to False.
            name (str, optional): Possible name of filter. Defaults to "Filter".
        """
        super().__init__(name=name)

        # set the class buffer parameters
        self.n_frames = n_frames
        self.n_in_chan = n_in_chan
        self.n_out_chan = n_out_chan
        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_size = pad_size
        self.buffer_size = window_size + (n_frames - 1) * hop_size

        # select the best fft/ifft/init functions
        self.is_real = is_real
        self.fft = jnp.fft.fft
        self.ifft = jnp.fft.ifft
        self.buffer_init = complex_zeros
        self.n_freq = window_size + pad_size
        if self.is_real:
            self.fft = jnp.fft.rfft
            self.ifft = jnp.fft.irfft
            self.buffer_init = RandomNormal(mean=0, stddev=1e-9)
            self.n_freq = (window_size + pad_size) // 2 + 1

    @staticmethod
    @jit
    def step_input_buffer(buffer_input, in_buffer):
        buffer_size = in_buffer.shape[0]
        in_size = buffer_input.shape[0]
        keep_size = buffer_size - in_size

        # shift the buffer to the left
        in_buffer = in_buffer.at[:keep_size].set(in_buffer[in_size:])

        # fill in the right with the new samples
        in_buffer = in_buffer.at[keep_size:].set(buffer_input)
        return in_buffer

    def buffer_input(self, buffer_input, buffer_name):
        in_buffer = hk.get_state(
            buffer_name,
            shape=[self.buffer_size, buffer_input.shape[-1]],
            init=self.buffer_init,
        )

        # circle rotate buffer with new samples at right
        in_buffer = self.step_input_buffer(buffer_input, in_buffer)

        hk.set_state(buffer_name, in_buffer)
        return in_buffer

    def stft_analysis(self, x, window, window_size, hop_size, pad_size, n_frames):
        window_idx = jnp.arange(window_size)[None, :]
        frame_idx = jnp.arange(n_frames)[:, None]
        window_idxs = window_idx + frame_idx * hop_size

        # index the buffer with the map and window
        windowed_x = x[window_idxs] * window[None, :, None]

        # 0 is T, 1 will be F, 2 is channels
        return self.fft(windowed_x, axis=1, n=window_size + pad_size)

    def buffered_stft_analysis(self, x, analysis_window, buffer_name="in_buffer_1"):
        buffer_x = self.buffer_input(x, buffer_name)
        cur_stft_frames = self.stft_analysis(
            buffer_x,
            analysis_window,
            self.window_size,
            self.hop_size,
            self.pad_size,
            self.n_frames,
        )
        return cur_stft_frames

    @staticmethod
    @jit
    def step_output_buffer(buffer_input, out_buffer):
        window_size = buffer_input.shape[0]
        overlap_size = out_buffer.shape[0]
        hop_size = window_size - overlap_size

        # shift buffer to the left
        out_buffer = out_buffer.at[:-hop_size].set(out_buffer[hop_size:])

        # zero out the rightmost samples
        out_buffer = out_buffer.at[-hop_size:].set(0)

        # overlap add in the inputs
        out_buffer = out_buffer + buffer_input[hop_size:]

        return out_buffer

    def buffer_output(self, buffer_input, buffer_name):
        out_buffer = hk.get_state(
            buffer_name,
            shape=[self.window_size + self.pad_size - self.hop_size, self.n_out_chan],
            init=self.buffer_init,
        )

        # constuct current output with overlap of buffer and new input
        cur_output = buffer_input[: self.hop_size] + out_buffer[: self.hop_size]

        # update buffer with shift and overlap add
        out_buffer = self.step_output_buffer(buffer_input, out_buffer)

        hk.set_state(buffer_name, out_buffer)
        return cur_output  # jnp.concatenate((cur_output, out_buffer), axis=0)

    # TODO right non buffered STFT synthesis
    def stft_synthesis(self, x, synthesis_window, buffer_name="out_buffer_1"):
        # inverse fft and synthesis window
        x_td = self.ifft(x, axis=0) * synthesis_window[:, None]

        # overlap add with buffer to get output and update buffer
        return self.buffer_output(x_td, buffer_name)

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("Buffer")
        parser.add_argument("--n_frames", type=int, default=1)
        parser.add_argument("--n_in_chan", type=int, default=1)
        parser.add_argument("--n_out_chan", type=int, default=1)
        parser.add_argument("--window_size", type=int, default=512)
        parser.add_argument("--hop_size", type=int, default=256)
        parser.add_argument("--pad_size", type=int, default=0)
        parser.add_argument("--is_real", action="store_true")
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "n_frames",
            "n_in_chan",
            "n_out_chan",
            "window_size",
            "hop_size",
            "pad_size",
            "is_real",
        ]
        return {k: kwargs[k] for k in keys}

    @abstractmethod
    def __call__(self):
        pass


class OverlapSave(BufferedFilterMixin, hk.Module):
    """The OLS base class. Override this and write your own __init__, add_args, and grab_args, and __ols_call__. The inputs and outputs of __ols_call are manged automatically such that the ouput dictionary value with key "out" is managed. All others are just stacked and can be accessed in the optimizer via the feature container."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_window = None
        self.w_init = complex_zeros

    def get_filter(self, name, shape=None):
        # assumes you want a buffer size filter
        if shape is None:
            shape = [self.n_frames, self.n_freq, self.n_in_chan]
        w = hk.get_parameter(name, shape, init=self.w_init, dtype=jnp.complex64)

        # time domain antialias
        w_td = (
            self.ifft(w, axis=1)
            .at[:, (self.window_size + self.pad_size) // 2 :, :]
            .set(0.0)
        )
        return self.fft(w_td, axis=1)

    def __call__(self, metadata=None, **kwargs):
        # collect buffers for all inputs
        kwargs_buffer = {
            k: self.buffered_stft_analysis(v, self.analysis_window, k)
            for k, v in kwargs.items()
        }

        # call the users filtering function
        out = self.__ols_call__(metadata=metadata, **kwargs_buffer)

        # only ols first model output
        if isinstance(out, dict):
            # get non time-domain aliased samples
            out["out"] = self.ifft(out["out"], axis=0)[-self.hop_size :]
            return out
        else:
            out = self.ifft(out, axis=0)[-self.hop_size :]
            return out

    @staticmethod
    def add_args(parent_parser):
        return super(OverlapSave, OverlapSave).add_args(parent_parser)

    @staticmethod
    def grab_args(kwargs):
        return super(OverlapSave, OverlapSave).grab_args(kwargs)

    @abstractmethod
    def __ols_call__(self):
        pass


class OverlapAdd(BufferedFilterMixin, hk.Module):
    """The OLA base class. Override this and write your own __init__, add_args, and grab_args, and __ola_call__. The inputs and outputs of __ola_call are manged automatically such that the ouput dictionary value with key "out" is managed.  All others are just stacked and can be accessed in the optimizer via the feature container."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_window = None
        self.synthesis_window = None
        self.w_init = complex_zeros

    # this function should not be called in the forward pass
    # it is from the 1984 Griffin, Lim IEEE paper
    # "Signal Estimation from Modified Short-Time Fourier Transform"
    def get_synthesis_window(self, analysis_window):
        N = len(analysis_window)
        slide = (N - 1) // self.hop_size

        # init as full overlap
        analysis_window_sq = analysis_window ** 2
        denom = analysis_window_sq

        # add in the shifts
        for i in range(1, slide + 1):
            # add in the front overlap
            denom = denom.at[: -i * self.hop_size + N].add(
                analysis_window_sq[i * self.hop_size - N :]
            )

            # add in the back overlap
            denom = denom.at[i * self.hop_size :].add(
                analysis_window_sq[: -i * self.hop_size]
            )

        return analysis_window / denom

    def get_filter(self, name, shape=None):
        if shape is None:
            # assumes you want a buffer size filter
            shape = [self.n_frames, self.n_freq, self.n_in_chan]

        w = hk.get_parameter(name, shape, init=self.w_init)
        return w

    def __call__(self, metadata=None, **kwargs):
        # collect buffered inputs for all inputs
        kwargs_buffer = {
            k: self.buffered_stft_analysis(v, self.analysis_window, k)
            for k, v in kwargs.items()
        }

        # call the users filtering function
        out = self.__ola_call__(metadata=metadata, **kwargs_buffer)

        # only ols first model output
        if isinstance(out, dict):
            # get non overlap added time-domain signal
            out["out"] = self.stft_synthesis(out["out"], self.synthesis_window)

            return out
        else:
            return self.stft_synthesis(out, self.synthesis_window)

    @staticmethod
    def add_args(parent_parser):
        return super(OverlapAdd, OverlapAdd).add_args(parent_parser)

    @staticmethod
    def grab_args(kwargs):
        return super(OverlapAdd, OverlapAdd).grab_args(kwargs)

    @abstractmethod
    def __ola_call__(self):
        pass


def make_inner_grad(filter, inner_fixed, frame_loss):
    """Functio to make the feature extractor for the filter when using grad.

    Args:
        filter (_type_): The filter from Haiku
        inner_fixed (_type_): Filter kwargs
        frame_loss (_type_): The AF or frame loss

    Returns:
        _type_: Function that computes the output values and gradients.
    """

    @jit
    def filter_wrapper_loss(filter_p, filter_s, cur_data_samples, metadata, key):
        out, filter_s = filter.apply(
            filter_p,
            filter_s,
            key,
            **cur_data_samples,
            **inner_fixed["filter_kwargs"],
            metadata=metadata
        )

        loss = frame_loss(out, cur_data_samples, metadata)
        return loss, (out, filter_s)

    return jax.value_and_grad(filter_wrapper_loss, has_aux=True)
