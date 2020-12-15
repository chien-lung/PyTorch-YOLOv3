#!/bin/bash
# Download GTSDB
wget -P GTSDB/ https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip
wget -P GTSDB/ https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TestIJCNN2013.zip
unzip TrainIJCNN2013.zip -d GTSDB/
unzip TestIJCNN2013.zip -d GTSDB/
# Download GTSRB
wget -P GTSRB/ https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
wget -P GTSRB/ https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Training_Images.zip -d GTSRB/
unzip GTSRB_Final_Test_Images.zip -d GTSRB/
mkdir signs
cp -r GTSRB/Final_Training/Images/ signs/
wget -P signs/ https://www.dropbox.com/s/d3zzdt5j836z4o9/GT_all.csv
# Download LISA
wget -P LISA/ http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip
unzip signDatabasePublicFramesOnly.zip -d LISA/
