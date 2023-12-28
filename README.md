# Driver Drowsiness Detection using Haar Cascade Algorithm in Deep Learning

## Overview

This project is a real-time driver drowsiness detection system implemented using the Haar Cascade algorithm. Drowsy driving is a significant safety concern that can lead to accidents on the road. This project aims to address this issue by utilizing computer vision techniques to detect signs of driver drowsiness and alert the driver in a timely manner.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Run the Program](#run-the-program)
  - [Configuration](#configuration)
- [Contributions](#contributions)


## Features

- **Face Detection:** The system uses the Haar Cascade classifier to detect the face of the driver in real-time.
  
- **Eye Detection:** Once the face is detected, the algorithm identifies the eyes within the face region using Haar Cascade classifiers.

- **Drowsiness Detection:** By monitoring the driver's eye closure and blink patterns, the system determines the level of drowsiness.

- **Alert Mechanism:** When signs of drowsiness are detected, the system triggers an alert to notify the driver. This could be in the form of an audible alarm, visual alert, or a combination of both.

- **Configurability:** Users can easily configure sensitivity thresholds and alert mechanisms to suit their preferences.

## Technologies Used

- **OpenCV:** The computer vision library is used for face and eye detection.

- **Haar Cascade Classifiers:** Pre-trained classifiers are employed for face and eye detection.

- **Python:** The entire system is implemented in Python for ease of understanding, modification, and extension.

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/SHAMSUNDAR20/driver-drowsiness-detection.git
