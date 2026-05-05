import kagglehub

# Download latest version
path = kagglehub.dataset_download("mssjss/fastmathxassumption-2025-handwritten-math-to-latex")

print("Path to dataset files:", path)