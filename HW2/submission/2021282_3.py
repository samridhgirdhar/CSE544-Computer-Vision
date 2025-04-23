#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install opencv-python numpy')


# In[1]:


import cv2
import numpy as np
import json
import glob


# In[2]:


CHECKERBOARD = (8, 6) 


# In[3]:


image_files = sorted(glob.glob("chessboard_dataset/*.jpeg"))


# In[ ]:


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


objpoints = []  # 3d points in the world coordinate system
imgpoints = []  # 2d points in the image plane

for fname in image_files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        
        corners_subpix = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        
        
        objpoints.append(objp)
        imgpoints.append(corners_subpix)
        
        
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_subpix, ret)
        


# In[ ]:


img_shape = cv2.imread(image_files[0]).shape
h, w = img_shape[:2]


ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None
)

print("Calibration was successful?", ret)  # ret is the overall RMS re-projection error
print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coeffs:\n", distCoeffs)


# In[8]:


print("Mean reprojection error = ", ret)


# In[9]:


rotation_mats = []
for rvec in rvecs:
    R, _ = cv2.Rodrigues(rvec)
    rotation_mats.append(R)


# In[ ]:


import json



k1 = distCoeffs[0, 0]
k2 = distCoeffs[0, 1]
p1 = distCoeffs[0, 2]  # tangential
p2 = distCoeffs[0, 3]  # tangential
k3 = distCoeffs[0, 4]


radial_coeffs = [ float(k1), float(k2), float(k3) ]


fx = cameraMatrix[0, 0]
fy = cameraMatrix[1, 1]
skew = cameraMatrix[0, 1] 
cx = cameraMatrix[0, 2]
cy = cameraMatrix[1, 2]



all_errors = []
for i in range(len(objpoints)):
    
    corners2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs
    )
    
    diff = corners2.reshape(-1,2) - imgpoints[i].reshape(-1,2)
    err = np.sqrt((diff**2).sum(axis=1))  
    all_errors.extend(err.tolist())
mean_error = np.mean(all_errors)
std_dev = np.std(all_errors)





extrinsics_list = []
for i, (R, t) in enumerate(zip(rotation_mats, tvecs), start=1):
    
    R_list = R.tolist()
    t_list = t.ravel().tolist()  
    extrinsics_list.append({
        "image_id": i,
        "rotation_matrix": R_list,
        "translation_vector": t_list
    })


calib_data = {
    "intrinsic_parameters": {
        "focal_length": [float(fx), float(fy)],
        "skew": float(skew),
        "principal_point": [float(cx), float(cy)]
    },
    "extrinsic_parameters": extrinsics_list,
    "radial_distortion_coefficients": radial_coeffs,
    "reprojection_errors": {
        "mean_error": mean_error,
        "std_dev": std_dev  
    }
}

with open("2021282_parameters.json", "w") as f:
    json.dump(calib_data, f, indent=4)


# In[11]:


import os

output_folder = "undistorted_samples"
os.makedirs(output_folder, exist_ok=True)

for i, fname in enumerate(image_files[:5], start=1):
    img = cv2.imread(fname)
    undistorted = cv2.undistort(img, cameraMatrix, distCoeffs)
    outname = os.path.join(output_folder, f"undistorted_{i}.jpg")
    cv2.imwrite(outname, undistorted)


# “After applying the estimated radial distortion coefficients, we undistorted five raw images. Because the lens in our device is not strongly wide‐angled, the geometric change near the edges is relatively subtle. Nonetheless, closer inspection (especially near image corners) shows slightly straighter horizontal/vertical lines compared to the original images.”

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt



per_image_errors = []

for i in range(len(objpoints)):

    projected_points, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i],
        cameraMatrix, distCoeffs
    )
    
    projected_points = projected_points.reshape(-1, 2)
    detected_points = imgpoints[i].reshape(-1, 2)

    
    diff = projected_points - detected_points
    errors = np.linalg.norm(diff, axis=1)  
    mean_error = np.mean(errors)
    per_image_errors.append(mean_error)


mean_error_all = np.mean(per_image_errors)
std_error_all = np.std(per_image_errors)

print("Per-image errors:", per_image_errors)
print("Mean Reprojection Error = ", mean_error_all)
print("Std Deviation of Error = ", std_error_all)


plt.figure(figsize=(10,5))
plt.bar(range(1, len(per_image_errors)+1), per_image_errors, color='steelblue')
plt.xlabel('Image Index')
plt.ylabel('Reprojection Error (pixels)')
plt.title('Per-Image Reprojection Error')
plt.show()


# In[ ]:


import os

overlay_folder = "corner_overlays"
os.makedirs(overlay_folder, exist_ok=True)

for i in range(len(objpoints)):
    img = cv2.imread(image_files[i])
    
    projected_points, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i],
        cameraMatrix, distCoeffs
    )
    projected_points = projected_points.reshape(-1,2)

    
    detected_points = imgpoints[i].reshape(-1,2)


    for (dx, dy) in detected_points:
        cv2.circle(img, (int(dx), int(dy)), 5, (0,255,0), -1)

    
    for (px, py) in projected_points:
        cv2.circle(img, (int(px), int(py)), 3, (0,0,255), -1)

    out_name = os.path.join(overlay_folder, f"corner_overlay_{i+1}.jpg")
    cv2.imwrite(out_name, img)


# “The re-projection error is computed by taking each detected 2D corner (x_det, y_det), projecting its corresponding 3D point using the estimated camera parameters to get (x_proj, y_proj), and measuring their Euclidean distance. This error is then averaged across all corners in an image to give a per-image metric, and further averaged across all images to get a global mean.”

# In[ ]:


import numpy as np

plane_normals_camera = []

for i, R in enumerate(rotation_mats):
    
    normal_cam = R[:, 2]  
    
    normal_cam = normal_cam / np.linalg.norm(normal_cam)
    plane_normals_camera.append(normal_cam)

    print(f"Image {i+1}: plane normal in camera frame = {normal_cam}")

