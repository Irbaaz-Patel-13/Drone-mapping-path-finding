import cv2
import numpy as np
import os
import json
import time

# Define input and output directories
input_dir = '/Users/irbaazpatel/PycharmProjects/Rl-capstone/input-images'
base_output_path = '/Users/irbaazpatel/PycharmProjects/Rl-capstone/Path-planning-output'
drone_image_path = "/Users/irbaazpatel/PycharmProjects/Rl-capstone/top-drone.png"

# Load drone image and resize it for visibility
drone_img = cv2.imread(drone_image_path)
drone_img = cv2.resize(drone_img, (40, 40))  # Make the drone image larger for clarity

# Ensure the base output directory exists
os.makedirs(base_output_path, exist_ok=True)

# Loop through each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        input_image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(base_output_path, os.path.splitext(filename)[0])
        os.makedirs(output_path, exist_ok=True)

        # Load and process the image
        image = cv2.imread(input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 270, 350)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area to remove noise
        min_contour_area = 100
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        origin = (50, 50)  # Example starting point
        visited_centroids = []
        path_sequence = []

        start_time = time.time()

        # Find and visit centroids using a Greedy approach
        while len(visited_centroids) < len(filtered_contours):
            unvisited_centroids = [
                (int(cv2.moments(c)['m10'] / cv2.moments(c)['m00']),
                 int(cv2.moments(c)['m01'] / cv2.moments(c)['m00']))
                for c in filtered_contours if (int(cv2.moments(c)['m10'] / cv2.moments(c)['m00']),
                                               int(cv2.moments(c)['m01'] / cv2.moments(c)[
                                                   'm00'])) not in visited_centroids
            ]

            if not unvisited_centroids:
                break

            # Greedy algorithm: Find the nearest centroid
            next_centroid = min(unvisited_centroids,
                                key=lambda p: np.sqrt((origin[0] - p[0]) ** 2 + (origin[1] - p[1]) ** 2))
            visited_centroids.append(next_centroid)
            path_sequence.append(next_centroid)
            path_sequence.append(origin)  # Return to origin after visiting each centroid

        time_taken = time.time() - start_time

        # Save the final path in a JSON file compatible with ArduPilot
        json_output_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_path.json")
        with open(json_output_path, 'w') as json_file:
            path_data = {
                "path": [
                    {
                        "latitude": point[0],  # x coordinate as latitude
                        "longitude": point[1]  # y coordinate as longitude
                    }
                    for point in path_sequence
                ]
            }
            json.dump(path_data, json_file, indent=4)

        # Video creation with clearer drone image and lines from origin to each centroid
        height, width, _ = image.shape
        video_output_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_path.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, 5, (width, height))

        # Annotate path on the image and save video frames
        for point in path_sequence:
            frame = image.copy()
            cv2.circle(frame, origin, 8, (255, 255, 0), -1)  # Highlight origin
            cv2.line(frame, origin, point, (0, 255, 255), 2)  # Draw line from origin to centroid

            # Overlay the drone image at the centroid
            x, y = point
            x_end = min(x + drone_img.shape[1], frame.shape[1])
            y_end = min(y + drone_img.shape[0], frame.shape[0])
            frame[y:y_end, x:x_end] = drone_img[0:y_end - y, 0:x_end - x]

            # Label each centroid
            cv2.putText(frame, f"Centroid: ({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(frame)

        out.release()

        # Save annotated output image with path sequence
        final_image_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_final.png")
        for i, point in enumerate(path_sequence):
            cv2.putText(image, f"{i + 1}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.line(image, origin, point, (0, 255, 0), 2)
        cv2.imwrite(final_image_path, image)

        # Print output summary for each image
        print(f"Processed {filename}")
        print("All outputs saved in:", output_path)
        print("Number of contours identified:", len(visited_centroids))
        print("Path sequence followed by the agent:", path_sequence)
        print("Closest contour coordinates visited each step:", visited_centroids)
        print("Total time taken:", time_taken, "seconds")
        print("Path data saved in:", json_output_path)
