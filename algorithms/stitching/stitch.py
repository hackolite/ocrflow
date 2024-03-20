import cv2
import numpy as np




def stich(image1, image2):
		# Load the image

		# Convert images to grayscale
		gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

		# Initialize the feature detector and extractor (e.g., SIFT)
		sift = cv2.SIFT_create()

		# Detect keypoints and compute descriptors for both images
		keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
		keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

		# Initialize the feature matcher using brute-force matching
		bf = cv2.BFMatcher()

		# Match the descriptors using brute-force matching
		matches = bf.match(descriptors1, descriptors2)

		# Select the top N matches
		num_matches = 50
		matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

		# Extract matching keypoints
		src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
		dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

		# Estimate the homography matrix
		homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

		# Warp the first image using the homography
		result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

		# Blending the warped image with the second image using alpha blending
		alpha = 0.5  # blending factor
		blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

		return blended_image