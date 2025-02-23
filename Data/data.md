# Directory

```
Download the datasets and structure them as shown below:

📦Dataset
 ┣ 📂Celeb-DF
 ┃ ┣ 📂Celeb-real
 ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┣ 📂images
 ┃ ┃ ┗ 📂videos
 ┃ ┣ 📂Celeb-synthesis
 ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┣ 📂images
 ┃ ┃ ┗ 📂videos
 ┃ ┣ 📂Youtube-real
 ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┣ 📂images
 ┃ ┃ ┗ 📂videos
 ┃ ┗ 📂splits
 ┣ 📂DeeperForensics
 ┃ ┣ 📂manipulated_sequences
 ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┣ 📂images
 ┃ ┃ ┗ 📂videos
 ┃ ┗ 📂splits
 ┣ 📂FF++
 ┃ ┣ 📂manipulated_sequences
 ┃ ┃ ┣ 📂Deepfakes
 ┃ ┃ ┃ ┣ 📂c0
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┣ 📂c23
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┗ 📂c40
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┣ 📂Face2Face
 ┃ ┃ ┃ ┣ 📂c0
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┣ 📂c23
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┗ 📂c40
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┣ 📂FaceShifter
 ┃ ┃ ┃ ┣ 📂c0
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┣ 📂c23
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┗ 📂c40
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┣ 📂FaceSwap
 ┃ ┃ ┃ ┣ 📂c0
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┣ 📂c23
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┗ 📂c40
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┗ 📂NeuralTextures
 ┃ ┃ ┃ ┣ 📂c0
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┣ 📂c23
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┃ ┗ 📂c40
 ┃ ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┗ 📂videos
 ┃ ┣ 📂original_sequences
 ┃ ┃ ┣ 📂c0
 ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┣ 📂c23
 ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┗ 📂videos
 ┃ ┃ ┗ 📂c40
 ┃ ┃ ┃ ┣ 📂bounding_boxes
 ┃ ┃ ┃ ┣ 📂facial_landmarks
 ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┗ 📂videos
 ┃ ┗ 📂splits
 ┗ 📂ForgeryNet
 ┃ ┣ 📂bounding_boxes
 ┃ ┣ 📂facial_landmarks
 ┃ ┣ 📂images
 ┃ ┣ 📂splits
 ┃ ┗ 📂videos
```

# Download links
[FaceForensics](https://github.com/ondyari/FaceForensics)  
[Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics)  
[DeeperForensics](https://github.com/EndlessSora/DeeperForensics-1.0)
[ForgeryNet](https://github.com/yinanhe/ForgeryNet)
