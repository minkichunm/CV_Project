import cv2
import numpy as np
import os, datetime

# Run params
PIC_EXTENSION = '.png'
dir_path = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = dir_path + '/input_data/'
OUTPUT_DIR = dir_path + '/output/'

# Algo params
KNN_MATCHING = True                    # If true, BFMatcher uses KNN with filtering (Lowe's ratio test), if False it will use best match
MIN_MATCH_COUNT = 10                    # How many points need to be matched to proceed with calculating homography

BEST_MATCH_DISTANCE_THRESHOLD = 3      # Only matches with distance below this threshold will be taken

KNN_DISTANCE_RATIO = 0.03               # KNN-matching Lowe's test ratio


# In order for the script to work,
def main():
    no_pics = len(os.listdir(INPUT_DIR))
    for x in range(1, no_pics):
        print('Stitching iteration' + str(x))
        stitch_two(x)


def stitch_two(x):
    if x == 1:
        left_color = cv2.imread(INPUT_DIR + str(x) + str(PIC_EXTENSION))
        left_bw = cv2.cvtColor(left_color,cv2.COLOR_BGR2GRAY)
    else:
        left_color = cv2.imread(OUTPUT_DIR + 'output_image_cropped_' + str(x - 1) + str(PIC_EXTENSION))
        left_bw = cv2.cvtColor(left_color,cv2.COLOR_BGR2GRAY)

    right_color = cv2.imread(INPUT_DIR + str(x+1) + str(PIC_EXTENSION))
    right_bw = cv2.cvtColor(right_color,cv2.COLOR_BGR2GRAY)

    # Finding key points with SIFT algorithm
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left_bw,None)
    kp2, des2 = sift.detectAndCompute(right_bw,None)

    # Draw keypoints
    cv2.imwrite(OUTPUT_DIR + 'left_keypoints_' + str(x) + str(PIC_EXTENSION), cv2.drawKeypoints(left_color,kp1,None))
    cv2.imwrite(OUTPUT_DIR + 'right_keypoints_' + str(x) + str(PIC_EXTENSION), cv2.drawKeypoints(right_color, kp1, None))

    # Match keypoints between left and right picture
    # KNN matching + filtering by distance (DISTANCE_RATIO global param)
    if KNN_MATCHING:
        matches = match_knn(des1, des2)
    else:
        matches = match_best(des1, des2)

    # Draw matching parameters
    draw_params = dict(matchColor=(0,255,0),
                        singlePointColor=None,
                        flags=2)

    matches_image = cv2.drawMatches(left_color,kp1,right_color,kp2,matches,None,**draw_params)
    cv2.imwrite(OUTPUT_DIR + 'matches_' + str(x) + str(PIC_EXTENSION), matches_image)
 
    if len(matches) >= MIN_MATCH_COUNT:
        # Converting interest points list to use as arguments for homography function
        dst_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        # Using RANSAC algorithm to find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h,w = right_bw.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        result_photo = cv2.perspectiveTransform(pts, M)
        overlapping_image = cv2.polylines(left_bw,[np.int32(result_photo)],True,255,3, cv2.LINE_AA)

        # Output overlapping image
        cv2.imwrite(OUTPUT_DIR + 'overlapping_image_' + str(x) + str(PIC_EXTENSION), overlapping_image)
    else:
        print("Not enought matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))

    #Warp right image on left image with homography matrix
    result_photo = cv2.warpPerspective(right_color,M,(left_color.shape[1] + right_color.shape[1], left_color.shape[0]))
    result_photo[0:left_color.shape[0],0:left_color.shape[1]] = left_color

    # Result image
    cv2.imwrite(OUTPUT_DIR + 'output_image_' + str(x) + str(PIC_EXTENSION), result_photo)

    # Trim the unalligned black frames 
    trimmed_result = trim(result_photo)

    # Output stitched picture
    cv2.imwrite(OUTPUT_DIR + 'output_image_cropped_' + str(x) + str(PIC_EXTENSION), trimmed_result)


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def match_knn(des1, des2):
    BF_matcher = cv2.BFMatcher()
    matches = BF_matcher.knnMatch(des1,des2,k=2)

    # Filter the matched points with Lowe's ratio test
    filtered_matches = []
    for first,second in matches:
        if first.distance <  KNN_DISTANCE_RATIO * second.distance:
            filtered_matches.append(first)

    return filtered_matches

def match_best(des1, des2):
    BF_matcher = cv2.BFMatcher()
    matches = BF_matcher.match(des1,des2)

    filtered_matches = []
    for match in matches:
        if match.distance < BEST_MATCH_DISTANCE_THRESHOLD:
            filtered_matches.append(match)

    return filtered_matches


if __name__=="__main__":
   main()