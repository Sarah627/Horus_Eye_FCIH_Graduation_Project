# AI-Powered Tourism Website (Horus Eye)
## Graduation Project Faculty Of Computers and Atrificial Intelligence Helwan University



<img src= "https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white">


![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![YAML](https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=openjdk&logoColor=white)

![Spring](https://img.shields.io/badge/spring-%236DB33F.svg?style=for-the-badge&logo=spring&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)


<br>

<div align="center">
  <img src="Images/Horus%20Eye.jpg">
</div>


## Table of Contents:
1. Overview
   1. Problem Statement.
   2. Proposed Solution.
2. Challenges
   1. General constraints.
   2. Data.
   3. Choosing Best Model.
3. How Did We Overcomed The Challenges Of The Dataset 
4. Results
   1. Results from YOLOv7
   2. How the project was deployed
5. Future Work
   1. Advanced Models
6. Acknowledgements 
7. References

--- 
## Overview:
### Problem Statement:
Egypt's civilization’s wealth is reflected in its historical landmarks and astonishing destinations. Exploring Egypt’s diverse cultural heritage can be a complex experience for international tourists. Typical guidebooks and online resources often fail to provide the real-time assistance and tailored insights needed to truly appreciate the wonders of Egypt. As a result, significant issues arise: tourists become uncertain about the historical value or significance of landmarks due to the sheer magnitude of options available and frustrated by the lack of seamless accommodation booking options. Additionally, the missed opportunities to experience unique journeys in Egypt are another considerable issue.

![tourism](Images/tourism.png)


Our project aims to address key barriers to Egypt's tourism industry:

1. **`Limited Landmark Information`**: Tourists often lack up-to-date details on Egypt’s landmarks, making it difficult to appreciate their cultural significance.
2. **`Inefficient Landmark Identification`**: Without guidance, tourists may struggle to identify and explore Egypt's cultural heritage.
3. **`Fragmented Booking Process`**: Booking accommodations is often disjointed, leading to confusion and frustration for travelers.
### Proposed Solution:

Our project aimed to convert the tourism experience for non-Egyptians, empowering them to explore Egypt's landmarks with confidence, ease, and greater enthusiasm, through the seamless integration of AI-enhanced image recognition technology and a user-friendly tourism platform.

Here’s a shortened version of the objectives:

#### Objectives
In line with our goal to enhance the travel experience for non-Egyptian tourists and improve their exploration of Egypt's landmarks, our project aims to:

1. **`Develop an AI Image Recognition System`**: Create an AI-powered system to identify landmarks in Egypt from user-uploaded photos, offering real-time insights.
2. **`Ensure Seamless User Experience`**: Design an intuitive website for easy photo uploads, landmark information navigation, and accommodation booking.
3. **`Provide Landmark Information`**: Curate a comprehensive database of Egypt’s landmarks, offering historical, cultural, and geographical insights to deepen users’ appreciation of Egypt’s heritage.

--- 

## Challenges:
### General Constraints
1. **Limited** access to high-quality datasets for training machine learning algorithms.
2. **Complexity** of integrating with external APIs such as booking.com or Trivago for accommodation bookings.
3. Challenges in developing accurate and reliable AI-enhanced image recognition models for identifying landmarks.
4. **Time constraints** due to academic deadlines for project completion.
5. Resource constraints such as **limited access to computational resources** or cloud services for model training and deployment.

### Data
Our dataset posed several significant challenges that complicated its use for object detection model training:
1. **`Lack of Annotations`**
   <details>

   1. The dataset consisted solely of raw images without any accompanying annotation files, CSV files, or metadata.
   2. This absence necessitated the manual creation of ground truth data, a process that is both time-consuming and labor-intensive.
    
   </detials>

2. **`Class Imbalance`**:
   <details>

   1. There was a notable imbalance in object classes, with some classes being significantly underrepresented.
   2. This imbalance risks producing a biased model that performs well on frequent classes but poorly on less frequent ones, affecting overall accuracy and reliability.
   </details> 
3. **`Presence of Outliers`**:
   <details>

   1. The dataset contained numerous outliers, such as low-quality images, irrelevant objects, and unusual object orientations.
   2. These outliers could introduce noise into the training process, potentially confusing the model and degrading its performance if not properly managed during preprocessing.
   </details> 
4. **`Lack of Preprocessing`**:
   <details>

   1. The dataset had not undergone any preprocessing, requiring extensive steps to ensure suitability for model training.
   2. Essential preprocessing tasks included resizing, normalization, and augmentation to improve the model's ability to generalize across varied input data.
   </details> 
Addressing these challenges was critical for developing a robust and accurate object detection system.

### Choosing Best Model

#### Model Selection Criteria for Object Detection

Selecting the right model for object detection involves balancing multiple factors:

- **`Accuracy`**: Measured using metrics like mAP and IoU, ensuring the model can reliably identify and localize objects.
- **`Speed`**: Important for real-time tasks, evaluated by inference time and throughput.
- **`Computational Requirements`**: Considerations include hardware needs, memory footprint, and energy efficiency.
- **`Model Complexity`**: Affects deployment and maintenance; simpler models are easier to deploy but might sacrifice performance.
- **`Adaptability`**: Ability to generalize to new tasks and data, using transfer learning and data augmentation.
- **`Support`**: Community resources, documentation, and ecosystem tools can streamline development.
- **Cost and Licensing**: Financial and legal aspects, particularly for commercial use.


#### Challenges and Limitations of Tested Models

1. **SSD**:
   - Struggles with small object detection and complex backgrounds.
   - Speed comes at the cost of accuracy.
   - Customization is labor-intensive.

2. **Faster R-CNN & Mask R-CNN**:
   - Complex architecture and long training times.
   - Issues with adapting to custom datasets, requiring significant adjustments to scripts.

3. **YOLOv7**:
   - Balances speed and accuracy, with an AP of 51.4% on our dataset.
   - Requires high hardware resources but performs well for real-time detection.

---

## How Did We Overcomed The Challenges Of The Dataset 

**`Selecting an Annotating Tool`**:  
After extensive research, we chose **Roboflow** for its user-friendly interface, YOLOv7 PyTorch compatibility, and free plan with multiple export formats.

**`Data Preprocessing with Roboflow`**:  
- **Resized** images to 640x640.  
- **Greyscaled** images to reduce complexity, as color is not crucial for object detection, improving model performance.

**`Addressing Class Imbalance`**:  
Used **data augmentation** to balance the dataset and enhance model robustness.

--- 


