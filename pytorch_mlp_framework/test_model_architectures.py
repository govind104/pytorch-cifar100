import unittest
import torch
from model_architectures import (
    BatchNormalizedConvolutionalProcessingBlock,
    BatchNormalizedConvolutionalDimensionalityReductionBlock,
    BatchNormalizedConvolutionalProcessingBlockWithResidualConnections,
)


class TestModelBlocks(unittest.TestCase):
    def setUp(self):
        # Common setup: input tensor with 8 samples, 16 channels, and 32x32 spatial dimensions
        self.input_tensor = torch.randn(8, 16, 32, 32)  # Batch=8, Channels=16, Height=Width=32
        self.num_filters = 16  # Match channels for compatibility in residual connections
        self.kernel_size = 3
        self.padding = 1
        self.bias = False
        self.dilation = 1
        self.reduction_factor = 2  # Downsample spatial dimensions by this factor

    def test_processing_block(self):
        """Test BatchNormalizedConvolutionalProcessingBlock for shape compatibility."""
        block = BatchNormalizedConvolutionalProcessingBlock(
            input_shape=self.input_tensor.shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation,
        )
        output = block(self.input_tensor)
        self.assertEqual(
            output.shape, self.input_tensor.shape,
            "Processing block should maintain the input shape"
        )

    def test_dimensionality_reduction_block(self):
        """Test BatchNormalizedConvolutionalDimensionalityReductionBlock for correct downsampling."""
        block = BatchNormalizedConvolutionalDimensionalityReductionBlock(
            input_shape=self.input_tensor.shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation,
            reduction_factor=self.reduction_factor,
        )
        output = block(self.input_tensor)
        expected_shape = (
            self.input_tensor.shape[0],  # Batch size (unchanged)
            self.num_filters,  # Number of filters (set by block)
            self.input_tensor.shape[2] // self.reduction_factor,  # Reduced height
            self.input_tensor.shape[3] // self.reduction_factor,  # Reduced width
        )
        self.assertEqual(
            output.shape, expected_shape,
            "Dimensionality reduction block failed to downsample correctly"
        )

    def test_processing_block_with_residual(self):
        """Test BatchNormalizedConvolutionalProcessingBlockWithResidualConnections for residual shape."""
        block = BatchNormalizedConvolutionalProcessingBlockWithResidualConnections(
            input_shape=self.input_tensor.shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation,
        )
        output = block(self.input_tensor)
        self.assertEqual(
            output.shape, self.input_tensor.shape,
            "Residual processing block should maintain the input shape"
        )


if __name__ == "__main__":
    unittest.main()