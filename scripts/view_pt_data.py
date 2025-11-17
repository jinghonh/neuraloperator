import torch
import os
import matplotlib.pyplot as plt

# Construct the absolute path to the data file
# The script is in 'scripts/', so we navigate relative to it.
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
file_path = os.path.join(project_root, 'data', 'navier_stokes', 'nsforcing_test_128.pt')

# Use torch.load() to load the data
try:
    data = torch.load(file_path)

    # Print some information about the data
    print(f"Successfully loaded data from: {file_path}")
    
    # Check the type of the loaded data
    print(f"Data type: {type(data)}")

    # If the data is a dictionary, print its keys and the shape of tensors
    if isinstance(data, dict):
        print("Data is a dictionary. Keys:", list(data.keys()))
        for key, value in data.items():
            print(f"\n--- Key: '{key}' ---")
            if isinstance(value, torch.Tensor):
                print(f"  Tensor shape: {value.shape}")
                print(f"  Tensor dtype: {value.dtype}")
            else:
                print(f"  Value type: {type(value)}")
                print(f"  Value preview: {str(value)[:100]}") # Print a preview
        
        # Visualize one sample
        if 'x' in data and 'y' in data:
            sample_index = 0
            sample_x = data['x'][sample_index]
            sample_y = data['y'][sample_index]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 'x' sample
            im_x = axes[0].imshow(sample_x, cmap='viridis')
            axes[0].set_title(f"Sample {sample_index} of 'x'")
            fig.colorbar(im_x, ax=axes[0])
            
            # Plot 'y' sample
            im_y = axes[1].imshow(sample_y, cmap='viridis')
            axes[1].set_title(f"Sample {sample_index} of 'y'")
            fig.colorbar(im_y, ax=axes[1])
            
            plt.suptitle("Data Visualization")
            plt.show()


    # If the data is a tensor, print its shape and dtype
    elif isinstance(data, torch.Tensor):
        print(f"Data is a tensor.")
        print(f"Tensor shape: {data.shape}")
        print(f"Tensor dtype: {data.dtype}")
    
    # You can uncomment the following line to print a preview of the data,
    # but be careful as it might be very large.
    # print("\nFull data preview:")
    # print(data)

except FileNotFoundError:
    print(f"Error: File not found. Please check the path: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

    print(f"An error occurred while loading the file: {e}")
