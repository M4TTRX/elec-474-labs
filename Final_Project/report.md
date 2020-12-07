# ELEC 474 Final Project
**Due Decemeber 8 2020**

Matthieu Roux - 20013052

Harry Chan - 20022216

--- 

## Declaration of Originality

We declare that we (Harry Chan and Matthieu Roux) are sole authors of the submitted solution. The code was developped over a shared repository and done in using pair programming methods over video call.

---

## Method used

As indicated in the project hints, a sheet of paper is a planar surface, while a face is not (I hope so at least). This means that it is possible to do a decent feature matching between the images triplets that use fake faces, while the feature matching between the real faces will not work as well.

In order for this to work properly we need to run the feature matching on faces only which requires some face detection method.

Our Training and Testing steps can be seen below

### Training

1. Import 70% of images by triplets
2. Detect and crop faces in each triplets
3. Select the two best faces in the triplets and perform feature matching and store the number of matches
4. repeat this process for all fake and real images
5. Declare a **match threshold** that is exactly between the highest number of matches in a real images, and the lowest amount of matches in a fake one

### Testing
1. Import the remaining 30% of images and add our own and put them in triplets
2. Detect and crop faces in each triplet
3. Select the two best faces in the triplets and perform feature matching and store the number of matches
4. Compare the number of matches with the **match threshold**: if there are **less matches** than in the threshold, the face is **real** if there are **more matches** than the threshold then the face is **fake**.

## Implementation
### Organising the data

We decided to leave around 30% of the data provided to us for testing our results, leaving 70% for training. The reason 30% was chosen is because it's a standard amouunt in testing sets.
All images were imported as a list of triplets, a triplet being a set of 3 images: the center one then its corresponding left and right image. This made it a lot easier for us to manipulate the data and ensure triplets would not get mixed up
### Face Detection

The face detection was done using the OpenCV `CascadeClassifier`. We noticed that there was no perfect Haar cascade model that would detect all faces so we decided to use all the ones that open cv could give us, until a face was found. This ensured that we would get at least 2 faces per triplet which is the minimum required to perform feature matching.

### Match Detection

Match detection was done by detecting keypoints using SIFT and amtching them using a Flann based matcher. We used the OpenCV `cv2.SIFT_create()`, `detectAndCompute` methods and the `cv2.FlannBasedMatcher().knnMatch` to obtain the pairs of bet matches for the two images on which matching was performed. Lowes filtering was used to filter out poor matches. Finally the total number of matches was returned.

When the face_finder would return two images, performing matching was easy: just use those two images. But sometimes the detector would inaccurately detect faces and return a total of 3 faces detected. We noticed that the inaccurate faces were usually smaller in image size than regular images so our fix was to perform matching on the two larger images only.

## Tests

Tests were performed on the 30% remaining data + our own faces. We created an evaluation matrix to compare results.

### Confusion Matrix

|                  | Predicted Positives | Predicted Negatives |
| :--------------- | ------------------- | ------------------- |
| Actual Positives |                     |                     |
| Actual Negatives |                     |                     |

**Precision** = 

**Recall** = 

**Training Time** = (x seconds per image triplet)

**Testing Time** = (x seconds per image triplet)

## Conclusion

Our system was very successful at predicting the provided data set and our own faces. It does have its flaws however and relies on some assumptions.

The solution is quite effective on tests that use faces on paper and that don't fold that paper, or wear a mask. If an imposter was to wrap the paper around their face, our solution would not work. Problems of this scope could perhaps be solved using a lidar add on.
Our solution is also unable to detect photoshoped faces.

We have not run into any computationally expensive processes because we used OpenCV's optimized libraries and used the python performance oriented practices (such as list comprehensions) to make sure our code runs smoothly.

If we had more time I am unsure on how we would improve the solution with the same restricted dataset, library and camera restrictions.

As previously mentioned, using a lidar camera would allow us to detect very effectively unusual 3d geometry like a sheet of paper in front of the face. With enough data we would have been able to train lidar data as to what an average face looks like, and to train a neural network to identify if the person is wearing a mask or not.
