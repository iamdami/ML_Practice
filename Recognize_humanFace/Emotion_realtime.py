from fer import Video
from fer import FER
import os
import sys
import pandas as pd

# location of the video file
location_videofile = " .mp4"

# Build the Face detection detector
face_detector = FER(mtcnn=True)
# Input the video
input_video = Video(location_videofile)

processing_data = input_video.analyze(face_detector, display=False)

# Convert the analysed information into a dataframe
vid_df = input_video.to_pandas(processing_data)
vid_df = input_video.get_first_face(vid_df)
vid_df = input_video.get_emotions(vid_df)

# Plotting the emotions against time
pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()

# Work on the dataframe to extract which emotion was prominent
angry = sum(vid_df.angry)
disgust = sum(vid_df.disgust)
fear = sum(vid_df.fear)
happy = sum(vid_df.happy)
sad = sum(vid_df.sad)
surprise = sum(vid_df.surprise)
neutral = sum(vid_df.neutral)

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]

score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
score_comparisons['Emotion Value from the Video'] = emotions_values
score_comparisons
