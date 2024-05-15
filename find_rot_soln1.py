import cv2
import numpy as np
import os


def find_rotation_angle(template_img, test_img):
    # Convert images to grayscale
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, firstLevel=0, WTA_K=2, patchSize=31)

    # Detect keypoints and compute descriptors
    keypoints_template, descriptors_template = orb.detectAndCompute(template_gray, None)
    keypoints_test, descriptors_test = orb.detectAndCompute(test_gray, None)

    # Initialize the BFMatcher with HAMMING distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Perform KNN matching
    nn_matches = matcher.knnMatch(descriptors_template, descriptors_test, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in nn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    match_img = cv2.drawMatches(template_img, keypoints_template, test_img, keypoints_test, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Extract locations of good matches
    matched_template_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matched_test_pts = np.float32([keypoints_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography matrix using RANSAC
    H, mask = cv2.findHomography(matched_template_pts, matched_test_pts, cv2.RANSAC, 5.0)

    # Calculate rotation angle from homography matrix
    if H is not None:
        rotation_angle = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
        return rotation_angle
    else:
        raise RuntimeError("Homography matrix could not be estimated.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to template and test image folders
    template_folder_name = "template_images"
    test_folder_name = "test_images"
    template_folder = os.path.join(script_dir,template_folder_name)
    test_folder = os.path.join(script_dir,test_folder_name)
    i = 1
    for template_filename in sorted(os.listdir(template_folder)):
        if template_filename.startswith("masked_"):
            template_path = os.path.join(template_folder, template_filename)
            template_img = cv2.imread(template_path)
            #cv2.imshow("Template_img",template_img)
            #cv2.waitKey(0)
            test_image_folder = os.path.join(test_folder, "type_" + str(i))
            # Iterate over each test image for this template
            for test_filename in sorted(os.listdir(test_image_folder)):
                if test_filename.startswith("test_"):
                    test_path = os.path.join(test_image_folder, test_filename)
                    test_img = cv2.imread(test_path)
                    ##cv2.imshow("Template_img",template_img)
                    ##cv2.waitKey(0)
                    # Find rotation angle
                    rotation_angle = find_rotation_angle(template_img.copy(), test_img.copy())

                    # Display results
                    cv2.putText(test_img, "Rotation Angle: {:.2f} degrees".format(rotation_angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    min_height = min(image.shape[0] for image in [template_img,test_img])
                    resized_images = [cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2))) for image in [template_img,test_img]]
                    concat_img = cv2.hconcat(resized_images)
                    cv2.imshow("Result", concat_img)
                    cv2.waitKey(0)

            i = i+1

    cv2.destroyAllWindows()
    
