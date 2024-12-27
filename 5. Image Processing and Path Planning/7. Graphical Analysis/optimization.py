import cv2
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define input and output directories
input_dir = '/Users/irbaazpatel/PycharmProjects/Rl-capstone/input-images'
base_output_path = '/Users/irbaazpatel/PycharmProjects/Rl-capstone/Path-planning-output'
drone_image_path = "/Users/irbaazpatel/PycharmProjects/Rl-capstone/top-drone.png"

# Load drone image and resize it for visibility
drone_img = cv2.imread(drone_image_path)
if drone_img is not None:
    drone_img = cv2.resize(drone_img, (40, 40))  # Make the drone image larger for clarity
else:
    print(f"Warning: Unable to load drone image from {drone_image_path}")

# Ensure the base output directory exists
os.makedirs(base_output_path, exist_ok=True)

# Initialize lists to store metrics for tables and graphs
metrics = {
    'filename': [], 'contour_count': [], 'min_contour_area': [],
    'max_contour_area': [], 'avg_contour_area': [], 'total_path_distance': [],
    'processing_time': []
}

# Loop through each image in the input directory
for idx, filename in enumerate(os.listdir(input_dir)):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Improved file extension check
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(base_output_path, f"yolo-output-image{idx + 1}.png")

        # Load and process the image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Warning: Unable to load image from {input_image_path}")
            # Append default values for metrics when the image is not loaded
            metrics['filename'].append(f"yolo-output-image{idx + 1}")
            metrics['contour_count'].append(0)
            metrics['min_contour_area'].append(0)
            metrics['max_contour_area'].append(0)
            metrics['avg_contour_area'].append(0)
            metrics['total_path_distance'].append(0)
            metrics['processing_time'].append(0)
            continue  # Skip to the next file if image cannot be loaded

        start_time = time.time()  # Start timing the processing

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 270, 350)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area to remove noise
        min_contour_area = 100
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        # Record contour metrics
        contour_areas = [cv2.contourArea(c) for c in filtered_contours]
        metrics['filename'].append(f"yolo-output-image{idx + 1}")
        metrics['contour_count'].append(len(filtered_contours))
        metrics['min_contour_area'].append(min(contour_areas) if contour_areas else 0)
        metrics['max_contour_area'].append(max(contour_areas) if contour_areas else 0)
        metrics['avg_contour_area'].append(np.mean(contour_areas) if contour_areas else 0)

        # Save the processed image with contours drawn
        cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)  # Draw contours in green
        cv2.imwrite(output_image_path, image)  # Save the processed image

        # Record processing time
        processing_time = time.time() - start_time
        metrics['processing_time'].append(processing_time)

# Ensure all lists in metrics have the same length
num_entries = len(metrics['filename'])
for key in metrics:
    if len(metrics[key]) != num_entries:
        print(f"Warning: {key} has a different length. Adjusting to match.")
        while len(metrics[key]) < num_entries:
            # Fill missing entries with zero or appropriate default values
            if key in ['contour_count', 'min_contour_area', 'max_contour_area', 'avg_contour_area', 'total_path_distance']:
                metrics[key].append(0)
            elif key == 'processing_time':
                metrics[key].append(0)

# Export metrics to a CSV file for further analysis
metrics_df = pd.DataFrame(metrics)
metrics_csv_path = os.path.join(base_output_path, "path_planning_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved to {metrics_csv_path}")

# Load the metrics from the CSV file
metrics_df = pd.read_csv(metrics_csv_path)

# Set the style of seaborn
sns.set(style="whitegrid", font_scale=1.5)  # Increase font size to avoid warnings

# Visualization 1: Bar chart for contour count
plt.figure(figsize=(12, 6))
sns.barplot(x='filename', y='contour_count', data=metrics_df, palette='viridis')
plt.title('Number of Contours Detected per Image', fontsize=16)
plt.xlabel('Image Filename', fontsize=14)
plt.ylabel('Contour Count', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, 'contour_count.png'))
plt.show()

# Visualization 2: Histogram of contour areas
plt.figure(figsize=(12, 6))
area_data = metrics_df[['min_contour_area', 'max_contour_area', 'avg_contour_area']].melt(var_name='Contour Type', value_name='Area')
sns.histplot(data=area_data, x='Area', hue='Contour Type', bins=30, kde=True, palette='Set1')
plt.title('Distribution of Contour Areas', fontsize=16)
plt.xlabel('Contour Area', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Contour Area Type', loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, 'contour_area_distribution.png'))
plt.show()

# Visualization 3: Line plot for total path distance
plt.figure(figsize=(12, 6))
sns.lineplot(x='filename', y='total_path_distance', data=metrics_df, marker='o')
plt.title('Total Path Distance per Image', fontsize=16)
plt.xlabel('Image Filename', fontsize=14)
plt.ylabel('Total Path Distance (pixels)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, 'path_distance.png'))
plt.show()

# Visualization 4: Bar chart for processing time
plt.figure(figsize=(12, 6))
sns.barplot(x='filename', y='processing_time', data=metrics_df, palette='coolwarm')
plt.title('Processing Time per Image', fontsize=16)
plt.xlabel('Image Filename', fontsize=14)
plt.ylabel('Processing Time (seconds)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, 'processing_time.png'))
plt.show()

# Additional visualizations for your report
# Visualization 5: Scatter plot for contour area vs. processing time
plt.figure(figsize=(12, 6))
sns.scatterplot(x='avg_contour_area', y='processing_time', data=metrics_df)
plt.title('Contour Area vs. Processing Time', fontsize=16)
plt.xlabel('Average Contour Area', fontsize=14)
plt.ylabel('Processing Time (seconds)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, 'contour_area_vs_processing_time.png'))
plt.show()

# Visualization 6: Box plot for contour count distribution
plt.figure(figsize=(12, 6))
sns.boxplot(x='contour_count', data=metrics_df)
plt.title('Distribution of Contour Counts', fontsize=16)
plt.xlabel('Contour Count', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, 'contour_count_distribution.png'))
plt.show()
