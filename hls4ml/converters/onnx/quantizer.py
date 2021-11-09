"""
Quantizer for the Quant node, after scale and zeropoint hafe been extracted. (Thus at this point they are 1 and 0.)

This is based on the sample implementation in finn-base
"""

import numpy as np
from hls4ml.model.hls_model import Quantizer

class QuantNodeQuantizer(Quantizer):
    """ This implements a quantizer for a FixedPrecisionType with width==integer"""
    def __init__(self, precision):
        assert(precision.width == precision.integer)
        super().__init__(precision.width, precision)

    def __call__(self, data):
        """ Apply the quantization on the data """
        # Clamping
        min_int_val = self._min_int(self.hls_type.signed, self.hls_type.saturation_mode, self.bits)
        max_int_val = self._max_int(self.hls_type.signed, self.bits)
        data = np.where(data > max_int_val, max_int_val, data)
        data = np.where(data < min_int_val, min_int_val, data)
        # Rounding
        rounding_fx = self._resolve_rounding_mode(self.hls_type.rounding_mode)
        return rounding_fx(data)


    @staticmethod
    def _min_int(signed: bool, saturation_mode: str, bit_width: int) -> int:
        """Compute the minimum integer representable by a given number of bits.
        Args:
            signed (bool): Indicates whether the represented integer is signed or not.
            saturation_mode (bool): Indicates the saturation mode used (AP_SAT_SYM or AP_SAT)
            bit_width (int): Number of bits available for the representation.
        Returns:
            int: Maximum unsigned integer that can be represented according to
            the input arguments.
        Examples:
            >>> min_int(signed=True, saturation_mode='AP_SAT_SYM', bit_width=8)
            int(-127)
            >>> min_int(signed=False, saturation_mode='AP_SAT_SYM', bit_width=8)
            int(0)
            >>> min_int(signed=True, saturation_mode='AP_SAT', bit_width=8)
            int(-128)
            >>> min_int(signed=False, saturation_mode='AP_SAT_SYM', bit_width=8)
            int(0)
        """
        if saturation_mode not in ("AP_SAT_SYM", "AP_SAT"):
            raise ValueError(f"Saturation mode {saturation_mode} not supported. Only AP_SAT_SYM, AP_SAT supported")
        if signed and saturation_mode == "AP_SAT_SYM":
            value = -(2 ** (bit_width - 1)) + 1
        elif signed:
            value = -(2 ** (bit_width - 1))
        else:
            value = 0
        return value

    @staticmethod
    def _max_int(signed: bool, bit_width: int) -> int:
        """Compute the maximum integer representable by a given number of bits.
        (Note, narrow and unsigned is not supported by the implementation, so saturation mode is not used)
        Args:
            signed (bool): Indicates whether the represented integer is signed or not.
            bit_width (int): Number of bits available for the representation.
        Returns:
            Tensor: Maximum integer that can be represented according to
            the input arguments.
        Examples:
            >>> max_int(signed=True, bit_width=8)
            int(127)
            >>> max_int(signed=False, bit_width=8)
            int(255)
        """
        if not signed:
            value = (2 ** bit_width) - 1
        else:
            value = (2 ** (bit_width - 1)) - 1
        return value

    @staticmethod
    def _resolve_rounding_mode(mode_string):
        """Resolve the rounding mode string of Quant and Trunc ops
        to the corresponding numpy functions."""
        if mode_string == "AP_RND_CONV":
            return np.round
        # elif mode_string == "CEIL":   # not supported
        #     return np.ceil
        elif mode_string == "AP_TRN":
            return np.floor
        else:
            raise ValueError(f"Could not resolve rounding mode called: {mode_string}")