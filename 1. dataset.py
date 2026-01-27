import kagglehub

# Download latest version
path = kagglehub.dataset_download("hearfool/vggface2")

print("Path to dataset files:", path)