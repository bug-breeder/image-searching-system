import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_results_from_json(json_file_path):
    """
    Read results from a JSON file and create visualizations for metrics.

    Parameters:
    json_file_path (str): Path to the JSON file containing results

    Returns:
    str: Output filename of the saved image
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        results = json.load(file)

    # Determine the metric type (Recall or mAP)
    metric_type = 'Recall' if 'Recall' in results[0] else 'mAP'

    # Extract unique metric levels
    metric_levels = list(results[0][metric_type].keys())

    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'{metric_type} Results - {os.path.basename(json_file_path)}', fontsize=16)

    # Flatten the axes for easier iteration
    axs_flat = axs.flatten()

    # Iterate through metric levels
    for i, level in enumerate(metric_levels):
        # Prepare data for plotting
        alphas = [result['Alpha'] for result in results]
        betas = [result['Beta'] for result in results]
        metrics = [result[metric_type][level] for result in results]

        # Create scatter plot
        scatter = axs_flat[i].scatter(alphas, metrics, c=betas, cmap='viridis',
                                      s=100, edgecolors='black', linewidth=1)

        # Customize the plot
        axs_flat[i].set_title(f'{metric_type} at Level {level}')
        axs_flat[i].set_xlabel('Alpha')
        axs_flat[i].set_ylabel(f'{metric_type} Value')

        # Add colorbar to show beta values
        plt.colorbar(scatter, ax=axs_flat[i], label='Beta')

        # Add grid for better readability
        axs_flat[i].grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Generate output filename
    base_filename = os.path.splitext(os.path.basename(json_file_path))[0]
    output_filename = f'{base_filename}_{
        metric_type.lower()}_visualization.png'

    # Save the figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Image saved to {output_filename}")

    # Close the plot to free up memory
    plt.close(fig)

    return output_filename


def process_json_files():
    """
    Process all JSON files in the current directory
    """
    # Get current directory
    current_dir = os.getcwd()

    # Find all JSON files
    json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]

    # Process each JSON file
    for json_file in json_files:
        try:
            plot_metric_results_from_json(os.path.join(current_dir, json_file))
        except Exception as e:
            print(f"Error processing {json_file}: {e}")


# Run the batch processing
process_json_files()
