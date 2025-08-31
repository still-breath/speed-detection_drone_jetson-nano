import tensorrt as trt

# Initialize the TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load the ONNX model
onnx_file = "simplified_model.onnx"  # Change this to your ONNX model path

# Create a TensorRT builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()

# Parse the ONNX model
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(onnx_file, "rb") as f:
    parser.parse(f.read())

# Enable FP16 precision using the builder's set_flag method
builder.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision

# Set the workspace size
builder.max_workspace_size = 1 << 30  # Set the workspace size (e.g., 1GB)

# Build the engine
engine = builder.build_cuda_engine(network)

# Save the TensorRT engine
engine_file = "opset10_malam_fp16.engine"  # Path to save the engine
with open(engine_file, "wb") as f:
    f.write(engine.serialize())

print("FP16 TensorRT engine created successfully!")
