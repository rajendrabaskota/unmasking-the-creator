# **Unmasking The Creator**
### This repository deals with classification of AI-generated and Human-crafted texts and images.

## **You can run the project locally by following these steps:**
   * Clone the repository to your local machine
   * Activate a virtual environment and run the command `pip install -r requirements.txt`
   * cd into backend directory
   * Run the command `uvicorn main:app --host 0.0.0.0 --port 8000` to start the backend server
   * Return back to the root directory of the project and cd into the frontend directory
   * Run the command `npm install` to install all the required dependencies for the frontend server
   * Run the command `npm start` to start the frontend server

---
## **Demo video for text and image classification**
https://github.com/rajendrabaskota/unmasking-the-creator/assets/66084649/4670647f-c0d8-4a79-830a-314171f4eaf9

---
## **1) AI Generated Text Detection**
### **Datasets Insights**
* #### **HC3-English dataset**
    * **Human-written texts - 24,300**
    * **AI-generated texts - 24,300**

* #### **GPT Wiki Intro dataset**
    * **Human-written texts - 150,000**
    * **AI-generated texts - 150,000**
---

### **Ongoing Tasks**
* #### **AI Dataset Generation using Llama-2-7B**
    * **Generated texts: 80,000**
    * **Goal: 200,000**
* #### **Model fine-tuining**
* #### **Exploration of different techniques**
---

### **Methodology**
The classification of text as AI-generated or human-crafted is based on 3 different methods:

* #### **Direct Text Classification using RoBERTa**
This method uses RoBERTa model which can directly classify whether the given text is written by an AI or a human. The model is fine-tuned with our above mentioned datasets.

* #### **Classification using Perplexity Method**
This method uses GPT-2 model to calculate perplexity of the given text. The calculated perplexity of the text is compared with the threshold value which is calculated from our dataset. The text having perplexity score lower than the threshold value is classified as AI-generated whereas the text is written by human if its perplexity is higher than the set threshold.

* #### **Domain-wise Perplexity Score Method**
The primary concept of this method is the same as the above method except that this method uses different thresholds for different texts based on the domain of the text. Thus, this method first classifies the given text into its respective domain using a fine-tuned RoBERTa model. The domains are based on the above mentioned datasets and the datasets are divided into 6 different domains: HC3 Reddit Eli5, HC3 Open QA, HC3 Medicine, HC3 Finance, HC3 Wiki CSAI and GPT-Wiki-Intro. After identifying the domain of the text, the perplexity of the given text is compared with the mean perplexity threshold of the respective domain. Finally, the text is classified as AI-generated or Human-crafted.
---

### **Results**
* Average F1 score based on perplexity score method: **90.67%**
---


## **2) AI Generated Image Detection**
### **Dataset Insights**
* #### **Training Dataset**
    * Taken from [this](https://arxiv.org/abs/1912.11035) work which contains 720,119 ProGAN real and fake images. The dataset can be found [here](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view) (dataset size ~ 70GB)
    * 80,000 images generated using ADM taking imagenet dataset as the real images are taken from [this](https://github.com/ZhendongWang6/DIRE) work.

* #### **Testing Dataset**
    * Testing is done on 20 different generative models
    * 12 different GAN models (~87,000 images) taken from [this](https://arxiv.org/abs/1912.11035) work
    * 8 different diffusion models (10,000 images) taken from [this](https://github.com/Yuheng-Li/UniversalFakeDetect) work
---
 
### **Methodology**
* Feature space is generated using CLIP:ViT-L/14
* The generated feature space is fed to several models like Logistic Regression, SVM, Neural Networks
* **Further plan**: Implementation of Teacher-Student modality

![image1](https://github.com/rajendrabaskota/unmasking-the-creator/assets/66084649/9cbe46db-03e0-4ca9-a54d-ef00338e3b68)

                              Block Diagram for AI Image Detection
---

### **Results**
#### The following results were obtained when trained using a Neural Network with input layer of 768 units and an output layer comprising of a single unit. The inputs to the network are the embeddings obtained from CLIP:ViT-L/14
* Trained using both ProGAN and ADM datasets
   * Learning Rate: 0.3
   * Epochs: 20k
   * 
     ![progan-adm](https://github.com/rajendrabaskota/unmasking-the-creator/assets/66084649/67193485-d03d-4629-9d35-8da62b89f163)
---

* Trained using only ProGAN dataset
   * Learning Rate: 0.03
   * Hidden Units: 100
   * Activation: ReLU
   * Dropout: 0.2
   * Epochs: 2500
   * 
     ![gan-only](https://github.com/rajendrabaskota/unmasking-the-creator/assets/66084649/5f365844-5444-41b0-b807-ba21c10c3f9c)
---

* Trained using only ADM dataset
   * Learning Rate: 0.03
   * Epochs: 3000
   *
     ![adm-only](https://github.com/rajendrabaskota/unmasking-the-creator/assets/66084649/278733e3-abc7-4c2e-98b5-92d0010d98fc)

---
