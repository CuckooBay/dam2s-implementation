url="http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/rgbd-scenes-v2_imgs.zip"
filename="rgbd-scenes-v2_imgs"
wget ${url}
unzip "rgbd-scenes-v2_imgs.zip"
rm "rgbd-scenes-v2_imgs.zip"
mv ${filename} "./data"
