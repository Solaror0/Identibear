import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

def resize_image(image, size=(37, 50)):
    """Resize an image to the specified size."""
    return cv2.resize(image, size)

def rotate_image_counter_clockwise(image):
    """Rotate an image 90 degrees counterclockwise."""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def mp4_to_jpg_with_labels(video_path, label=1, test_size=0.2):
    # Define directories under the current directory
    current_dir = os.getcwd()
    train_images_dir = os.path.join(current_dir, 'train', 'images')
    train_labels_dir = os.path.join(current_dir, 'train', 'labels')
    test_images_dir = os.path.join(current_dir, 'test', 'images')
    test_labels_dir = os.path.join(current_dir, 'test', 'labels')

    # Create output directories if they don't exist
    for dir_path in [train_images_dir, train_labels_dir, test_images_dir, test_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    total_frame_count = 0
    # Capture the video from the file
    for video in video_path:
        cap = cv2.VideoCapture(video)
        frame_count=0
        frames = []
        
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                break

            # Rotate the frame 90 degrees counterclockwise
            rotated_frame = rotate_image_counter_clockwise(frame)
            frames.append(rotated_frame)
            frame_count += 1

        # Release the video capture object
        cap.release()
        print(f"Extracted {frame_count} frames from {video_path}")
        total_frame_count+=frame_count
        # Split frames into training and testing sets
        train_frames, test_frames = train_test_split(frames, test_size=test_size, random_state=42)

        # Save frames and labels from the video
        for idx, frame in enumerate(train_frames):
            idx+=total_frame_count-frame_count
            resized_frame = resize_image(frame)  # Resize the frame to 37x50
            gray_frame = convert_to_grayscale(resized_frame)  # Convert to grayscale
            output_image_file = os.path.join(train_images_dir, f"{idx:04d}.jpg")
            output_label_file = os.path.join(train_labels_dir, f"{idx:04d}.txt")

            # Save the grayscale frame as a JPG file
            cv2.imwrite(output_image_file, gray_frame)

            # Save the corresponding label
            with open(output_label_file, 'w') as f:
                f.write(str(label))

        for idx, frame in enumerate(test_frames):
            idx+=total_frame_count-frame_count
            resized_frame = resize_image(frame)  # Resize the frame to 37x50
            gray_frame = convert_to_grayscale(resized_frame)  # Convert to grayscale
            output_image_file = os.path.join(test_images_dir, f"{idx:04d}.jpg")
            output_label_file = os.path.join(test_labels_dir, f"{idx:04d}.txt")

            # Save the grayscale frame as a JPG file
            cv2.imwrite(output_image_file, gray_frame)

            # Save the corresponding label
            with open(output_label_file, 'w') as f:
                f.write(str(label))
        label+=1
    print(f"Saved {len(train_frames)} frames to {train_images_dir} and {len(test_frames)} frames to {test_images_dir}")

def add_lfw_to_directories(test_size=0.2):
    # Fetch the LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=5, resize=0.4)

    # Define directories under the current directory
    current_dir = os.getcwd()
    train_images_dir = os.path.join(current_dir, 'train', 'images')
    train_labels_dir = os.path.join(current_dir, 'train', 'labels')
    test_images_dir = os.path.join(current_dir, 'test', 'images')
    test_labels_dir = os.path.join(current_dir, 'test', 'labels')

    # Get the number of samples
    n_samples = lfw_people.images.shape[0]

    # Print the number of samples fetched
    print(f"Total LFW samples: {n_samples}")

    # Generate a list of indices
    indices = np.arange(n_samples)

    # Split indices into training and testing sets
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

    # Save LFW images and labels to the training directory
    for idx in train_indices:
        output_image_file = os.path.join(train_images_dir, f"{n_samples + idx:04d}.jpg")
        output_label_file = os.path.join(train_labels_dir, f"{n_samples + idx:04d}.txt")

        # Resize the LFW image
        img = (lfw_people.images[idx] * 255).astype(np.uint8)
        resized_img = resize_image(img)  # Resize to 37x50
        cv2.imwrite(output_image_file, resized_img)
        
        # Check if the image was saved successfully
        if os.path.exists(output_image_file):
            print(f"Saved train image: {output_image_file}")
        else:
            print(f"Failed to save train image: {output_image_file}")

        # Save the corresponding label as '0'
        with open(output_label_file, 'w') as f:
            f.write('0')  # Set the label to 0

    # Save LFW images and labels to the testing directory
    for idx in test_indices:
        output_image_file = os.path.join(test_images_dir, f"{n_samples + idx:04d}.jpg")
        output_label_file = os.path.join(test_labels_dir, f"{n_samples + idx:04d}.txt")

        # Resize the LFW image
        img = (lfw_people.images[idx] * 255).astype(np.uint8)
        resized_img = resize_image(img)  # Resize to 37x50
        cv2.imwrite(output_image_file, resized_img)

        # Check if the image was saved successfully
        if os.path.exists(output_image_file):
            print(f"Saved test image: {output_image_file}")
        else:
            print(f"Failed to save test image: {output_image_file}")

        # Save the corresponding label as '0'
        with open(output_label_file, 'w') as f:
            f.write('0')  # Set the label to 0

    print(f"Saved LFW images to {train_images_dir} and {test_images_dir}")

# Example usage
def create_data(videos):

    mp4_to_jpg_with_labels(videos)
    add_lfw_to_directories()

videos = ['vid.mp4', 'vid2.mp4']

create_data(videos)
