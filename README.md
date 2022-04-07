# pyx_COV19
# A projet to detect COVID-19 from X-rays using Convolutional Neural Networks.
** A journey with artificial intelligence in service of medical diagnosis **

To see the app >> https://share.streamlit.io/valerie-ducret/pyx_cov19/main/xray_app.py

## Project's aims

Coronaviruses are a diverse group of viruses infecting many different animals, and they can cause mild to severe respiratory infections in humans (Hu et al., 2020). In late 2019, a novel coronavirus designated as SARS-Cov-2 emerged in the city of Wuhan (China) and caused an outbreak of unusual viral pneumonia, called COVID-19, that rapidly developed as an unprecedented world sanitary crisis. One year after, about 74 million people all other the world has contracted COVID-19, and about 1.7 million people have died from it. This highly contagious virus, by binding to epithelial cells in the respiratory tract, starts replicating and migrating down to the airways and enters alveolar epithelial cells in the lungs. Fast replication of SARS-CoV-2 in the lungs may trigger a strong immune response (Hu et al., 2020). Thus, cytokine storm syndrome causes acute respiratory distress syndrome and respiratory failure, which is considered the main cause of death in patients with COVID-19 (Huang et al., 2020; Mehta et al., 2020). Therefore, this outbreak has led to drastic human and economic consequences for which we still do not gauge final magnitude. Despite public health responses trying to decrease contamination rate, the rapidly evolving situation conducted to a saturation of hospitalization demands and an increase in mortality rate. 
	One efficient measure to contain the disease and delay the spread is active testing for COVID-19. By identifying contaminated people, public organization can take measures to isolate and limit contacts with the infected ones. Indeed, without testing, it is impossible to discriminate COVID-19 from other virus like influenza responsible for the flu, because symptoms are very similar depending on the severity of COVID-19. The gold-standard tests that detect viruses are RT-PCR (Real-Time Polymerase Chain Reaction). The high-sensitivity of PCR tests are almost 100% accurate in spotting infected people, when they are administered properly. But such tests generally require trained personnel, specific reagents that are lacking in periods of high demands, and expensive machines that take time to provide results. Therefore, other methods were developed to detect COVID-19 rapidly and efficiently. For instance, antigen assays are faster and cheaper than PCR tests, but are not as sensitive and could miss infectious people. Other alternative methods incorporating analysis of chest radiographies (computed tomography CT-scans and X-rays) may assist in identifying false negative RT-PCR cases or when RT-PCR tests are unavailable. However, it is tricky and time-consuming for radiologists to discriminate COVID-19 from other viral pneumonia by eye.
Today, computer vision is assisting an increasing number of doctors to better diagnose their patients, monitor the evolution of diseases, and prescribe the right treatments. It is an emerging field that takes advantage of artificial intelligence algorithms that process images and often make a faster and more accurate diagnosis than humans could do. Potential application of computer vision systems is minimizing false positives in the diagnostic process or detect the slightest presence of a condition. One difficulty for radiologists comes down to discriminating between a ?classical? viral pneumonia and pneumonia caused by COVID-19. A study demonstrates that radiologists had high specificity (true negative rate) but moderate sensitivity (true positive rate) in differentiating COVID-19 from viral pneumonia on chest CT-scans (Bai et al., 2020). Also, analyses of chest X-rays and CT-scans may show different sensitivities in detecting COVID-19. Despite that chest X-ray abnormalities of COVID-19 mirror those of CT-scans, less dense opacities may be more difficult to detect by eye and conduct to a sensitivity of 69% compared to generally more than 90% for CT-scans (Wong et al., 2020). Therefore, the use of such methodology could highly help mitigate the burden on the healthcare system by providing accurate models that detect COVID-19 on CT-scans (Ahuja et al., 2020) and particularly on chest X-rays that show moderate sensitivity.
The objective of this project is to build a multiclass classification model that can accurately predict COVID-19 from chest X-rays, and particularly discriminate from viral pneumonia or X-rays taking from healthy patients.

## Data
The initial data consist of 2905 images:
?	219 images of COVID-19 X-rays (which represent about 7.5% of total data)
?	1341 images of normal X-rays (which represent 46.2% of total data)
?	1345 images of viral pneumonia X-rays (which represent 46.3% of total data)
The initial data have a size of 1024x1024 pixels. Most of the images are in grayscale but few of them show blue hues.
Note: On December the 12th of 2020, the number of images in the COVID-19 category from Kaggle increased to 1143 and the images were resized to 256x256 pixels.

No issues were come across upon the study of the data. Although at first, when we proceeded to some preliminary analysis, we noticed that some images were duplicated. How to verify if two images are identical? The signature of an image resides in its array. For a computer, an image is perceived as an array or matrix of values ranging from 0 to 255. By comparing all the values of a matrix element by element to another matrix, we would check whether two images are identical.
We then decided not to pursue further with the idea of dropping duplicated images as we would if we had to deal with more structured data. The first reason is that this preliminary analysis was time consuming for a result that was not significant enough, that is to say that, in reality, less than 10 duplicated images were found throughout the Normal and Viral Pneumonia categories and less than 25 duplicated images were found in the COVID-19 category.
The second reason, as one may notice, comes from the fact that the data is unbalanced in favor of the main target variable which is the COVID-19 label. To compensate for the unbalanced dataset, two ideas were opted, namely data augmentation and adjusting class weights.
Data augmentation corresponds to a data analysis technique that is used to increase the amount of data by modifying copies of already existing data. These modifications consist of applying geometric transformations such as flipping, cropping, rotation etc. As for the second idea, we added more weight to the COVID-19 category so that the model puts much more emphasis during training. Otherwise, this category would have been missed out and the opportunity of detecting COVID-19 cases would have fallen short of expectations.

The data were split in two parts with respect to the proportion of data between classes:
?	The training set amounts to 80% of the total data
?	The test set amounts to 20% of the total data
Project

We started to manually construct a convolutional neural network (CNN) model that first extract the features from images using filters and then classify using a multi-layer perceptron. However, the model was surely too complex, the accuracy was particularly low and the model not robust. Therefore, we decided to use the transfer learning technique, that permits to reduce the pre-processing part and gain accuracy for classification. We decided to use a pre-trained VGG16 model on ImageNet as it achieves top-5 test accuracy. 
As far as the data augmentation technique is concerned, the following modifications were applied to the batch of images on the training and test set:
?	Rescaling pixel values between 0 and 1 by dividing them by 255
?	Random rotations with a range of 10 degrees
?	Random zoom with a range between 0.9 and 1.1
?	Width shift with a range between -0.1 and 0.1
?	Height shift with a range between -0.1 and 0.1
?	Horizontal flip
Only the horizontal flip was not applied to the batch of images in the test set.

The pre-trained VGG-16 model was imported for feature extraction and used as entries for the image classification, with the following layers:
?	Global Average Pooling 2D
?	Flatten
?	Dense with 256 units and ReLU activation function
?	Dropout with rate = 0.5
?	Dense with 128 units and ReLU activation function
?	Dropout with rate = 0.5
?	Dense with 64 units and ReLU activation function
?	Dropout with rate = 0.5
The output layer is a Dense layer with 3 units and a softmax activation function (multinomial probability distribution) for our multi-class classification problem. We noticed that the addition of dense layers (in forward decreasing units) allowed to gain classification accuracy.

The model was compiled with the Adam optimizer with an initial learning rate of 0.001, the loss function used was a sparse categorical cross-entropy and the metrics used were accuracy, AUC (Area Under the Receiver Operating Characteristic Curve) and F1-score. The model was fit using the adjusted class weights on 20 epochs. 
We then proceeded to fine-tuning the model by removing the last 10 layers of the pre-trained VGG16 model with an initial learning rate of 0.0001, adding a 'ReduceLROnPlateau' callback which aims for changing the learning rate with respect to a chosen metric on the test set. Here we chose to monitor the validation loss function, that is to say that the learning rate would be reduced when the quantity has stopped decreasing. The fine-tuned model was fit using the adjusted class weights on 30 epochs and the aforementioned callback.

Results and conclusion
Using the first dataset provided on Kaggle (i.e. unbalanced dataset with 219 COVID-19 images), we could already obtain a very accurate classification on the test set with 92% accuracy, 99% AUC and 92% F1-score. Also, the test loss is at 21%. By looking at the normalized confusion matrix, the recalls for COVID-19 and normal X-rays are almost perfect (respectively 98% and 96%), however the viral pneumonia category needs some improving as, for instance, 11% of images are predicted as normal.
Using the recent and improved dataset (i.e. 1143 COVID-19 images) and without fine-tuning, we could grab additional accuracy with 94% accuracy, 99% AUC and 94% F1-score. The test loss is at 18%. The recalls for viral pneumonia is better with 91% instead of 88% previously. Interestingly, the recalls for COVID-19 and normal X-rays are lower with respectively 96% instead of 98% regarding COVID-19 and 93% instead of 96% regarding normal X-rays. 
After fine-tuning, we obtain 98% accuracy, 100% AUC and 98% F1-score. The test loss drastically decreased to 6%. Also, we get a recall of 100% for COVID-19, 97% for normal X-rays (1% predicted COVID-19 and 3% predicted viral pneumonia) and 96% for viral pneumonia (4% predicted normal). Thus, the metrics are particularly good for all categories and the model has a 100% sensitivity for COVID-19.

Therefore, we could demonstrate the very efficient method of deep transfer learning for radiography analyses and diagnosis, with correct architecture and fine-tuning. There are improvements that can still be made to this project. One remaining problem is to understand how the model learned to classify the images for each category (model interpretability). By highlighting the most important features, we could potentially show the characteristics of contracting COVID-19 and those features could be used for further model exploration such as the evolution of the disease through time, or a comparison between population categories (age, gender, ...). 

The data is availaible on: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

## References 

Ahuja S., Panigrahi B.K., Dey N.�et al.�(2020). Deep transfer learning-based automated detection of COVID-19 from lung CT scan slices.�Appl Intell. https://doi.org/10.1007/s10489-020-01826-w.

Bai H., Hsieh B., Xiong Z. et al. (2020). Performance of radiologists in differentiating COVID-19 from viral pneumonia on chest CT. Radiology, doi:�10.1148/radiol.2020200823.

Dong E., Du H., Gardner L. (2020). An interactive web-based dashboard to track COVID-19 in real time. Lancet Infect Dis2020:S1473-3099(20)30120-1. doi:10.1016/S1473-3099(20)30120-1. pmid:32087114.

Hu B., Guo H., Zhou P.�et al.�(2020). Characteristics of SARS-CoV-2 and COVID-19.�Nat Rev Microbiol.�https://doi.org/10.1038/s41579-020-00459-7.

Huang C. et al. (2020). Clinical features of patients infected with 2019 novel coronavirus in Wuhan, China.�Lancet�395, 497?506.

Mehta P. et al. (2020). COVID-19: consider cytokine storm syndromes and immunosuppression.�Lancet�395, 1033?1034.

Wong H.Y.F., Lam H.Y.S., Fong A. H-T. et al. (2020). Frequency and distribution of chest radiographic findings in patients positive for COVID-19. Radiology 296(2), E78. 

