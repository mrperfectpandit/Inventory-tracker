# Inventory-tracker

## _Our mission to use Automatic Tracking system to reduce the workload of client.

<a target="_blank" href="www.linkedin.com/in/aman-sharma-01b185190/">
  <img align="left" alt="LinkdeIN" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />
</a>  <a target="_blank" href="https://www.instagram.com/aman___sharma/">
  <img align="left" alt="Instagram" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/instagram.svg" /></a>  <a target="_blank" href="mailto:aman.sharmatds1999@gmail.com"> 
  <img align="left" alt="Gmail" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/gmail.svg" />
</a>  <a target="_blank" href="https://portfolioaman.herokuapp.com/">
  <img align="left" alt="Devto" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/dev-dot-to.svg" />
</a>  
<br>
<br>

![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg) ![ForTheBadge makes-people-smile](http://ForTheBadge.com/images/badges/makes-people-smile.svg)  ![ForTheBadge built-with-swag](http://ForTheBadge.com/images/badges/built-with-swag.svg)

## Datasets 
[The data is in video format](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) <br>

## Table of Content
  * [Overview](#Overview)
  * [Motivation](#Motivation)
  * [Features](#Features)
  * [Samples Pictures or Demo](#Samples-Pictures-or-Demo)
  * [Video Demo](#Video-Demo)
  * [Installation and run](#Installation-and-run)
  * [Contributing](#Contributing)
  * [Directory Tree](#Directory-Tree)
  * [Tech](#Tech)
  * [Bug / Feature Request](#bug-feature-request)

## Overview
This Project use the YOLOv4 object detection Algorithm to track the Person and clothes and identify the no. of counts of Process clothes for each workstation.

## Approach

1. First we make about 200-300 frames of from the video data and make a training Dataset
2. use [labelImg](https://github.com/tzutalin/labelImg) for Annotation of data and save each file in YOLO format
    <img  align='center' src="https://www.researchgate.net/profile/Thi-Le-5/publication/337705605/figure/fig3/AS:831927326089217@1575358339500/Structure-of-one-output-cell-in-YOLO.ppm">
-->using LabelIMG we craete to categories 
   1. Throwing
   2. working
3.Use google colab for training our data-set. For training we use YOLOv4 custom object detector with Darkent in the Cloud
4.After training save the best_weights and export our model.
5.Than perform fine-tuning and optimization of model using ROI.
6.Make final output video along its dashboard.

## Result

![final_result_first_run](.mp4)

