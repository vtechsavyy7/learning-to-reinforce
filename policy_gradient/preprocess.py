# Utility functions for preprocessing the frames from the Space Invaders environment
import cv2
import numpy as np


def preprocess_frame(frame):
    """Preprocess the input frame by converting it to grayscale, resizing it to 84x84, and normalizing pixel values to [0, 1]
    
    Args:
        frame (numpy array): The input frame from the environment (210x160x3 RGB image)
    
    Returns:
        numpy array: The preprocessed frame (84x84 grayscale image with pixel values in [0, 1])
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def queue_new_frame(curr_frames, new_frame, is_new_episode):
    """Queue the new frame with the previous frames to create a state representation for the DQN agent
    
    Args:
        curr_frames (numpy array): A numpy array of shape (4, 84, 84) representing the previous 4 frames in the queue
        new_frame (numpy array): The new preprocessed frame to add to the queue
        is_new_episode (bool): Whether this is the start of a new episode (if True, the entire queue will be reset with the new frame)

    Returns:
        numpy array: The queued state representation (4x84x84)
    """

    if is_new_episode:
        # If it's a new episode, reset the queue with the new frame repeated 4 times
        new_queue = np.stack([new_frame] * 4, axis=0)
    else:
        # Otherwise, shift the existing frames and add the new frame to the end of the queue
        new_queue = np.zeros_like(curr_frames)
        new_queue[:-1] = curr_frames[1:]  # Shift frames to the left
        new_queue[-1] = new_frame  # Add new frame to the end of the queue

    return new_queue

def preprocess_and_queue(curr_frames, new_frame, is_new_episode):
    """Pre-process and queue a new frame
    
    Args: 

    """
    preprocessed_frame = preprocess_frame(new_frame)
    return queue_new_frame(curr_frames, preprocessed_frame, is_new_episode)
