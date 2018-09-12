#!/bin/bash
wget -O ./tasks/R2R/snapshots/release/selfplaynew_glove_sample_train_dec \
  http://people.eecs.berkeley.edu/~ronghang/projects/speaker_follower/models/selfplaynew_glove_sample_train_dec

wget -O ./tasks/R2R/snapshots/release/selfplaynew_glove_sample_train_enc \
  http://people.eecs.berkeley.edu/~ronghang/projects/speaker_follower/models/selfplaynew_glove_sample_train_enc

wget -O ./tasks/R2R/snapshots/release/panorama_glove_speaker_teacher_imagenet_mean_pooled_train_iter_15000_dec \
  http://people.eecs.berkeley.edu/~ronghang/projects/speaker_follower/models/panorama_glove_speaker_teacher_imagenet_mean_pooled_train_iter_15000_dec

wget -O ./tasks/R2R/snapshots/release/panorama_glove_speaker_teacher_imagenet_mean_pooled_train_iter_15000_enc \
  http://people.eecs.berkeley.edu/~ronghang/projects/speaker_follower/models/panorama_glove_speaker_teacher_imagenet_mean_pooled_train_iter_15000_enc
