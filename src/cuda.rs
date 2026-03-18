// CUDA implementation of the CNN using rustacuda

use crate::cnn::*;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    input_matrix: DeviceBox<InputMatrix>,
    conv_output: DeviceBox<ConvOutput>,
    output: DeviceBox<OutputVec>,
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        )?;

        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Pre-allocate device buffers (input first for deterministic behavior)
        let input_matrix = DeviceBox::new(&InputMatrix([[0.0; INPUT_DIM]; INPUT_DIM]))?;
        let conv_output = DeviceBox::new(&ConvOutput(
            [[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE],
        ))?;
        let output = DeviceBox::new(&OutputVec([0.0; OUT_LAYER_SIZE]))?;

        // Copy CNN weights to device (persistent across compute calls)
        let conv_layer = DeviceBox::new(&cnn.conv_layer)?;
        let output_layer = DeviceBox::new(&cnn.output_layer)?;

        Ok(CudaContext {
            input_matrix,
            conv_output,
            output,
            conv_layer,
            output_layer,
            module,
            stream,
            _context,
        })
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        // Copy input matrix from host to device
        self.input_matrix.copy_from(input)?;

        // Local refs to avoid borrow issues with launch! macro
        let module = &self.module;
        let stream = &self.stream;

        let block_size = 256u32;

        // Kernel 1 Convolution + ReLU 
        // 4000 threads total (10 filters * 20 * 20 output elements)
        let conv_total = (CONV_LAYER_SIZE * CONV_OUT_DIM * CONV_OUT_DIM) as u32;
        let grid_conv = (conv_total + block_size - 1) / block_size;

        unsafe {
            launch!(module.conv_relu<<<grid_conv, block_size, 0, stream>>>(
                self.input_matrix.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                self.conv_output.as_device_ptr()
            ))?;
        }

        // Kernel 2 Output layer with parallel reduction
        // 10 blocks (one per output neuron), 256 threads each
        let grid_output = OUT_LAYER_SIZE as u32;

        unsafe {
            launch!(module.output_layer<<<grid_output, block_size, 0, stream>>>(
                self.conv_output.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                self.output.as_device_ptr()
            ))?;
        }

        // Single synchronize: CUDA ops on the same stream serialize automatically
        stream.synchronize()?;

        // Copy result from device to host
        let mut result = OutputVec([0.0; OUT_LAYER_SIZE]);
        self.output.copy_to(&mut result)?;

        Ok(result)
    }
}
