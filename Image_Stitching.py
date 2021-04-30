import cv2
import numpy as np


class Image_Stitching():
    # def __init__(self) :

    def tutorial(self):
        img_ = cv2.imread('images/image_left.jpg')
        # img_ = cv2.resize(img_, (0, 0), fx=1, fy=1)
        img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

        img = cv2.imread('images/image_right.jpg')
        # img = cv2.resize(img, (0, 0), fx=1, fy=1)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        # find the key points and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        cv2.imshow('image_left_keypoints', cv2.drawKeypoints(img_, kp1, None))

        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)

        # FLANN matcher code
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        # match = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = match.knnMatch(des1, des2, k=2)

        # BFMatcher matcher code:
        match = cv2.BFMatcher()
        matches = match.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.03*n.distance:
                good.append(m)


        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        flags=2)

        img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)
        cv2.imshow("image_drawMatches.jpg", img3)
        cv2.waitKey(0)

        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            cv2.imshow("image_overlapping.jpg", img2)
            cv2.waitKey(0)
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

        dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
        dst[0:img.shape[0], 0:img.shape[1]] = img
        cv2.imshow("image_stiched_crop.jpg", dst)
        cv2.waitKey(0)

        cv2.imshow("image_stitched_crop.jpg", trim(dst))
        cv2.waitKey(0)


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

# Defining main function
def main():
    Image_Stitching().tutorial()


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
