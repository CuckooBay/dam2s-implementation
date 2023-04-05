obj="tomato soda_can mushroom lightbulb coffee_mug"
for i in ${obj}
do
    url="http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/${i}_1.tar"
    filename="rgbd-scenes-v2_${i}"
    wget ${url}
    unzip "${filename}.zip"
    rm "${filename}.zip"
    mv ${filename} "./data"
done
