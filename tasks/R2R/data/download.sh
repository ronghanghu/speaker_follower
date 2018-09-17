#!/bin/sh

wget https://storage.googleapis.com/bringmeaspoon/R2Rdata/R2R_train.json -P tasks/R2R/data/
wget https://storage.googleapis.com/bringmeaspoon/R2Rdata/R2R_val_seen.json -P tasks/R2R/data/
wget https://storage.googleapis.com/bringmeaspoon/R2Rdata/R2R_val_unseen.json -P tasks/R2R/data/
wget https://storage.googleapis.com/bringmeaspoon/R2Rdata/R2R_test.json -P tasks/R2R/data/
wget http://people.eecs.berkeley.edu/~ronghang/projects/speaker_follower/data_augmentation/R2R_data_augmentation_paths.json -P tasks/R2R/data/
