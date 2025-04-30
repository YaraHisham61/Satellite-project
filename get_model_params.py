from model_arch.cloud_net_deep_supervision import model_arch

from profiler.profile import profile

# Define input size (channels, height, width)
input_size = (1,4, 512, 512)

model = model_arch(num_of_channels=4, num_of_classes=1)
# Profile the model
num_ops, num_params = profile(model, input_size)

print(f"Number of Parameters: {num_params}")
print(f"Number of Operations (MACs): {num_ops}")