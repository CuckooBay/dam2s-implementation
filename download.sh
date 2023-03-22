url="http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/rgbd-scenes-v2_imgs.zip"
filename="rgbd-scenes-v2_imgs"
wget ${url}
rm "rgbd-scenes-v2_imgs.zip"
unzip "rgbd-scenes-v2_imgs.zip"
mv ${filename} "./data"