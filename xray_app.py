import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
import modules.appsession as session
from modules.img_classification import import_and_predict
from modules.features_map import select_and_features_map
from modules.grad_cam import get_img_array, make_gradcam_heatmap, display_grad_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def main():
    state = session._get_state()
    pages = {
        'Introduction' : page_introduction,
        'Data exploration & visualization' : page_eda,
        'Model building' : page_cnn,
        'Results & Interpretation' : page_results,
        'Prediction' : page_prediction,
        'Conclusion' : page_conclusion,
    }
    st.sidebar.title('pyX_CVD19')
    st.sidebar.subheader('Menu')
    page = st.sidebar.selectbox('', tuple(pages.keys()))
    pages[page](state)
    state.sync()
    st.sidebar.write(
        'DataScientest Project'
        '\n\n'
        'October 2020 Bootcamp Class'
        '\n\n'
        '<u>Authors:</u>'
        '\n\n'
        '<strong>Kévin PAME</strong> [linkedin](https://www.linkedin.com/in/k%C3%A9vin-pame-9a138914b/)'
        '\n\n'
        '<strong>Valérie DUCRET</strong> [linkedin](https://www.linkedin.com/in/val%C3%A9rie-ducret-104a3526/)', unsafe_allow_html=True)

# ###################
# Page Introduction #
# ###################

def page_introduction(state):
    st.markdown("<h1 style = 'text-align : center'>Detection of COVID-19 from chest X-Ray</h1>", unsafe_allow_html=True)
    img = Image.open('static/lungs_AI.jpg')
    st.image(img, use_column_width = True)
    st.write('\n\n')
    st.write(
        'In <em>late 2019</em>, a novel coronavirus designated as <strong>SARS-Cov-2</strong> emerged in the city of Wuhan (China) and caused an outbreak of unusual viral pneumonia called <strong>COVID-19</strong>. This disease rapidly developed as an unprecedented world sanitary crisis.'
        '\n\n'
        'One year after, about 74 million people has contracted COVID-19 all over the world and about 1.7 million people have died from it.'
        '\n\n'
        'Despite public health responses trying to decrease contamination rate, the rapidly evolving situation conducted to a saturation of hospitalization demands and an increase in mortality rate.'
        '\n\n'
        'One efficient action to contain the disease and delay the spread is active testing for COVID-19. The gold-standard tests that detect viruses are <strong>RT-PCR</strong> (<strong><em>Real-Time Polymerase Chain Reaction</em></strong>). But such tests generally require trained personnel, specific reagents that are lacking in periods of high demands, and expensive machines that take time to provide results.'
        '\n\n'
        'Today, <strong>computer vision</strong> is assisting an increasing number of doctors to better diagnose their patients, monitor the evolution of diseases, and prescribe the right treatments. It is an emerging field that takes advantage of <strong>artificial intelligence</strong> algorithms that process images and often make a faster and more accurate diagnosis than humans could do.'
        '\n\n'
        'The objective of this project is to build a <strong>multiclass classification model</strong> that can <strong>accurately predict COVID-19 from chest X-rays</strong>, and particularly discriminate from viral pneumonia or X-rays taken from healthy patients.', unsafe_allow_html = True)
    
# ####################################
# Page Data exploration & visualization #
# ####################################

def page_eda(state):
    st.title('Data exploration & visualization')
    st.header("Visualization of X-rays")

    if st.button("Load a subset of raw images"):
        
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            path_normal="dataset/NORMAL"
            files_normal=os.listdir(path_normal)
            img_normal=random.choice(files_normal)
            st.image(path_normal+'/'+img_normal, width=200, caption = "Healthy patient X-ray")
        with col2:
            path_viral="dataset/VIRAL"
            files_viral=os.listdir(path_viral)
            img_viral=random.choice(files_viral)
            st.image(Image.open(path_viral+'/'+img_viral), width=200, caption = "Viral pneumonia X-ray")
        with col3:
            path_covid="dataset/COVID"
            files_covid=os.listdir(path_covid)
            img_covid=random.choice(files_covid)
            st.image(Image.open(path_covid+'/'+img_covid), width=200, caption = "Covid-19 X-ray")
        
        st.info("It is clearly difficult, particularly for untrained people, to detect the presence of COVID-19. The difficulty also comes from the low quality of some X-rays.")
    st.write('\n\n')
    st.header('Initial dataset')
    st.write('\n\n')
    st.write(
        'The initial data consist of 2905 images:'
        '\n\n'
        '- <strong>219 images</strong> of <strong>COVID-19</strong> X-rays (which represent about <strong><em>7.5% of total data</em></strong>)'
        '\n\n'
        '- <strong>1341 images</strong> of <strong>normal</strong> X-rays (which represent <strong><em>46.2% of total data</em></strong>)'
        '\n\n'
        '- <strong>1345 images</strong> of <strong>viral pneumonia</strong> X-rays (which represent <strong><em>46.3% of total data</em></strong>)'
        '\n\n'
        'The initial data have a size of <strong>1024x1024 pixels</strong>. Most of the images are in grayscale but few of them show blue hues.'
        '\n\n'
        '<u>Note:</u> On <strong><em>December the 12th of 2020</em></strong>, the number of images in the COVID-19 category from <em>Kaggle</em> increased to 1143 and the images were resized to <strong>256x256 pixels</strong>.', unsafe_allow_html = True)
    
    st.header("Data distribution")

    table_dataset = pd.DataFrame({
        "COVID-19":[219, 1143],
        "Normal": [1341, 1341],
        "Viral Pneumonia": [1345, 1345],
        "Image size": ["1024 x 1024", "256 x 256"]
    }, index=["First dataset", "Second dataset"])

    # Display table
    st.write(table_dataset)

    # Display piechart
    if st.checkbox("Show Pie Charts"):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,12), constrained_layout=True)
        labels = 'COVID-19', 'Healthy patients', 'Viral pneumonia'
        size1 = [219, 1341, 1345]
        size2 = [1143, 1341, 1345]
        explode = (0.1, 0, 0)  # only "explode" the 1st slice (i.e. 'covid')
        ax1.pie(x = size1,
                autopct = "%.1f%%",
                explode = (0.2, 0, 0),
                labels = labels,
                pctdistance = 0.6,
                shadow=True,
                textprops = {'fontsize': 13},
        )
        ax2.pie(x = size2,
                autopct = "%.1f%%",
                explode = explode,
                labels = labels,
                pctdistance = 0.6,
                shadow=True,
                textprops = {'fontsize': 13},
        )
        st.pyplot(fig)

    st.write(
        'The initial dataset is strongly <strong>imbalanced</strong> in favor of the main target variable which is the COVID-19 label. To compensate for it, but also to obtain a robust model, two ideas were opted, namely <strong>adjusting class weights</strong> and <strong>data augmentation</strong>.'
        '\n\n'
        '<strong>Data augmentation</strong> corresponds to a data analysis technique that is used to increase the amount of data by modifying copies of already existing data. These modifications consist of applying <strong>geometric transformations</strong> such as <em>flipping</em>, <em>cropping</em>, <em>rotation</em> etc.'
        '\n\n'
        'As for the second idea, we added more <strong>weight</strong> to the COVID-19 category so that the model puts much more emphasis during training. Otherwise, this category would have been missed out and the opportunity of detecting COVID-19 cases would have fallen short of expectations.'
        '\n\n'
        'The data were split in two parts <strong>with respect to the proportion of data between classes</strong>:'
        '\n\n'
        '- The <strong>training set</strong> amounts to <strong><em>80%</em></strong> of the total data'
        '\n\n'
        '- The <strong>test set</strong> amounts to <strong><em>20%</em></strong> of the total data', unsafe_allow_html = True)
   
    st.header("Keras Image Generator")

    if st.checkbox('Try image generator on COVID X-ray'):
        rotation_range = st.slider('Rotation', -90, 90, 0)
        width_shift_range = st.slider('Width shift', -1.0, 1.0, 0.0)
        height_shift_range = st.slider('Height shift', -1.0, 1.0, 0.0)
        zoom_range = st.slider('Zoom', -10, 10, 0)
        shear_range = st.slider('Shear', -90, 90, 0)
        aug = ImageDataGenerator(rescale = 1./255,
                                rotation_range = rotation_range,
                                width_shift_range = width_shift_range,
                                height_shift_range = height_shift_range,
                                zoom_range = zoom_range,
                                shear_range = shear_range,
                                horizontal_flip = True,
                                fill_mode = "nearest")
        augmented = aug.flow_from_directory("dataset",
                                            target_size=(256,256),
                                            shuffle=True,
                                            class_mode="categorical",
                                            batch_size= 10,
                                            save_to_dir=None)
        x_augmented, y_augmented = next(augmented)
        cols = st.beta_columns(3)
        for i in range(3):
            with cols[i]:
                st.image(x_augmented[i], use_column_width=True)
        st.info("We can see that data augmentation with Keras performs random manipulations on images")
    
# #####################
# Page Model building #
# #####################

def page_cnn(state):
    st.title('CNN Modelling')
    st.write('\n\n')
    st.write(
        'To build our efficient <strong>Convolutional Neural Network</strong>, we used the <strong>transfer learning</strong> technique that allows us to benefit from a relatively small computation time, and thus to save training time, and weights of a <em>pre-trained</em> model to increase the algorithm performance. Finally, this technique does not need a huge amount of X-rays data to get really efficient prediction of COVID-19.'
        '\n\n'
        'Therefore, we imported a <strong>VGG16</strong> model pre-trained on ImageNet, froze the convolutional layers, and we built a classifier made of <strong>dense</strong> and <strong>dropout</strong> layers.', unsafe_allow_html = True)
    st.header('Architecture')
    st.write('\n\n')
    img = Image.open('static/VGG16_architecture.jpg')
    st.image(img, use_column_width = True)
    st.write(
        '\n\n'
        'Interestingly, the simple addition of dense layers with decreasing units after the convolution part was clearly increasing classification accuracy.')
    st.header('Training hyperparameters')
    st.write('\n\n')
    st.write(
        'The model was compiled with the <strong>Adam optimizer</strong> and an initial <strong>learning rate</strong> of <em>0.001</em>.'
        '\n\n'
        'The <strong>loss function</strong> used was a <strong>sparse categorical cross-entropy</strong> and the <strong>metrics</strong> used were <strong>accuracy</strong>, <strong>AUC</strong> (<strong><em>Area Under the Receiver Operating Characteristic Curve</em></strong>) and a <strong>F1-score</strong> that we defined using <em>recall</em> and <em>precision</em>.'
        '\n\n'
        'The model was fit using the adjusted class weights on <em>20 epochs</em>. '
        '\n\n'
        'We then proceeded to fine-tuning the model by unfreezing the last 10 layers of the pre-trained VGG16 model with an initial learning rate of <em>0.0001</em>, adding a “<strong>ReduceLROnPlateau</strong>” callback which aims for changing the learning rate with respect to a chosen metric on the test set.'
        '\n\n'
        'Here we chose to monitor the <strong>validation loss function</strong>, that is to say that the learning rate would be reduced when the quantity has stopped decreasing.'
        '\n\n'
        'The fine-tuned model was fit using the adjusted class weights on <em>30 epochs</em> and the aforementioned callback.', unsafe_allow_html = True)
    
    with st.beta_expander('Visualize the feature maps during convolution'):
        st.write('\n\n')
        st.write(
            'Convolutional Neural Network models have impressive <strong>classification performance</strong>. Yet it is not clear <em>why</em> they perform so well and thus <em>how</em> they might be improved.'
            '\n\n'
            'Those models use filters performing the <em>convolution operation</em> and result in <strong>activation maps</strong> or <strong>features maps</strong>. These <strong>activation maps</strong> capture the result of applying filters to an input such as the input image or another feature map. Both filters and feature maps can be visualized.'
            '\n\n'
            'For instance, we can try to understand small filters, such as <em>contour or line detectors</em>. Using feature maps that result from the filtering, we may even get insight into the <strong>internal representation</strong> that the model has of a particular input.'
            '\n\n'
            'The idea of visualizing a feature map for a specific input image would be to understand what features of the input are detected in the feature maps.', unsafe_allow_html = True)
        st.write("")
        if st.checkbox("Generate a feature map"):
            category=st.selectbox("Choose the category", ("<select>","COVID","NORMAL","VIRAL PNEUMONIA"))
            if category !="<select>":
                st.write("You selected", category)
                waiting_text = st.text("Please wait...")
                if category=="COVID":
                    path_covid="dataset/COVID"
                    files_covid=os.listdir(path_covid)
                    img_covid=random.choice(files_covid)
                    image = path_covid+'/'+img_covid
                    n_layer = st.select_slider("Convolution layer", [1,2,4,5,7,8,9,11,12,13,15,16,17])
                    select_and_features_map(image, n_layer = n_layer)
                    waiting_text.text("")
                    st.info(
                        'You are visualizing 64 feature maps as subplots. Those maps were generated from the selected convolutional having a COVID-19 X-ray as input.'
                        '\n\n'
                        'The feature maps ***close to the input*** detect **small** or **fine-grained details**, whereas feature maps ***close to the output*** of the model capture more **general features**.'
                        '\n\n'
                        'We can see that the result of applying filters in the first convolutional layer is a lot of versions of X-ray with different features highlighted. For example, some highlight lines, other focus on the background or the foreground.'
                        '\n\n'
                        'Bright areas are the ***activated*** regions, meaning the filter detected the pattern it was looking for.'
                        '\n\n'
                        'You can move the slider to see the feature maps of the other convolution layers (number 17 corresponds to the last one).')
                elif category=="NORMAL":
                    path_normal="dataset/NORMAL"
                    files_normal=os.listdir(path_normal)
                    img_normal=random.choice(files_normal)
                    image = path_normal+'/'+img_normal
                    n_layer = st.select_slider("Convolution layer", [1,2,4,5,7,8,9,11,12,13,15,16,17])
                    select_and_features_map(image, n_layer = n_layer)
                    waiting_text.text("")
                    st.info(
                        "You are visualizing 64 feature maps as subplots. Those maps were generated from the selected convolutional layer having a healthy patient's X-ray as input."
                        '\n\n'
                        'The feature maps ***close to the input*** detect **small** or **fine-grained details**, whereas feature maps ***close to the output*** of the model capture more **general features**.'
                        '\n\n'
                        'We can see that the result of applying filters in the first convolutional layer is a lot of versions of X-ray with different features highlighted. For example, some highlight lines, other focus on the background or the foreground.'
                        '\n\n'
                        'Bright areas are the ***activated*** regions, meaning the filter detected the pattern it was looking for.'
                        '\n\n'
                        'You can move the slider to see the feature maps of the other convolution layers (number 17 corresponds to the last one).')
                else:
                    path_viral="dataset/VIRAL"
                    files_viral=os.listdir(path_viral)
                    img_viral=random.choice(files_viral)
                    image = path_viral+'/'+img_viral
                    n_layer = st.select_slider("Convolution layer", [1,2,4,5,7,8,9,11,12,13,15,16,17])
                    select_and_features_map(image, n_layer = n_layer)
                    waiting_text.text("")
                    st.info(
                        'You are visualizing 64 feature maps as subplots. Those maps were generated from the selected convolutional layer having a viral pneumonia X-ray as input.'
                        '\n\n'
                        'The feature maps ***close to the input*** detect **small** or **fine-grained details**, whereas feature maps ***close to the output*** of the model capture more **general features**.'
                        '\n\n'
                        'We can see that the result of applying filters in the first convolutional layer is a lot of versions of X-ray with different features highlighted. For example, some highlight lines, other focus on the background or the foreground.'
                        '\n\n'
                        'Bright areas are the ***activated*** regions, meaning the filter detected the pattern it was looking for.'
                        '\n\n'
                        'You can move the slider to see the feature maps of the other convolution layers (number 17 corresponds to the last one).')
        
# ###############################
# Page Results & Interpretation #
# ###############################

def page_results(state):
    st.title('Results')
    st.write('\n\n')
    st.warning('Given that two different datasets were used since the last update on *Kaggle*, two *identical* models were trained on the images. Here we will show you the results of both models and compare them as far as metrics are concerned. Yet, with our fine-tuned model, we could achieve a 100% accuracy to detect COVID from Chest X-rays of the testing set.')
    st.write('\n\n')
    with st.beta_expander('Display results for the first model'):
        st.subheader('Model trained on the initial dataset (*imbalanced dataset*)')
        st.write('\n\n')
        st.info('In medicine and biological fields, having a **low false negative rate** is the priority since making decisions upon tests with higher false negative rates or smaller recalls could be lethal.')
        st.write('Here, we created a <em>normalized</em> confusion matrix in such a way that only the recalls are shown.', unsafe_allow_html = True)
        st.write(
            '\n\n'
            '<u>Normalized confusion matrix</u>', unsafe_allow_html = True)
        cm_img = Image.open('static/VGG16_first_model_heatmap_confusion_matrix.jpg')
        st.image(cm_img, use_column_width = True)
        st.write(
            '\n\n'
            '<u>Classification report</u>', unsafe_allow_html = True)

        classification_firstmodel = pd.DataFrame({
            "precision":[0.91,0.89,0.97," "," ", 0.92, 0.93],
            "recall": [0.98, 0.96, 0.88," "," ", 0.94, 0.93],
            "f1-score": [0.95, 0.93, 0.92," ",0.93, 0.93, 0.93],
            "support": [44, 268,269, " ", 581, 581, 581]
        }, index=["COVID-19", "NORMAL", "VIRAL", " ", "accuracy", "macro avg", "weighted avg"])

        # Display classification report
        st.write(classification_firstmodel)

        st.write('\n\n')
        st.write(
            '\n\n'
            'We obtained an accurate classification on the test set with <strong>93% accuracy</strong>, <strong>99% AUC</strong> and <strong>93% F1-score</strong>.'
            '\n\n'
            'The <strong>recalls</strong> for <em>COVID-19</em> and <em>normal</em> X-rays are almost perfect (respectively <strong>98%</strong> and <strong>96%</strong>), however the <em>viral pneumonia</em> category needs some improving as, for instance, 11% of images are predicted as normal.', unsafe_allow_html = True)
        st.subheader('Fine-tuned model')
        st.write('\n\n')
        st.write(
            '\n\n'
            '<u>Normalized confusion matrix</u>', unsafe_allow_html = True)
        ft_cm_img = Image.open('static/VGG16_finetuned_first_model_heatmap_confusion_matrix.jpg')
        st.image(ft_cm_img, use_column_width = True)
        st.write(
            '\n\n'
            '<u>Classification report</u>', unsafe_allow_html = True)

        classification_firstmodel_tuned = pd.DataFrame({
            "precision":[0.98,0.93,0.99," "," ", 0.97, 0.96],
            "recall": [0.98, 1.00, 0.93," "," ", 0.97, 0.96],
            "f1-score": [0.98, 0.96, 0.96," ",0.96, 0.97, 0.96],
            "support": [44, 268,269, " ", 581, 581, 581]
        }, index=["COVID-19", "NORMAL", "VIRAL", " ", "accuracy", "macro avg", "weighted avg"])
        # Display classification report
        st.write(classification_firstmodel_tuned)
        
        st.write('\n\n')
        st.write(
            '\n\n'
            'Regarding the fine-tuned model, we obtained a very accurate classification on the test set with <strong>96% accuracy</strong>, <strong>99% AUC</strong> and <strong>96% F1-score</strong>.'
            '\n\n'
            'The <strong>recalls</strong> for <em>COVID-19</em> and <em>normal</em> X-rays are almost perfect (respectively <strong>98%</strong> and <strong>100%</strong>), only the <em>viral pneumonia</em> category has a smaller recall, yet it is still important with a high value of <strong>93%</strong>. Only 7% of images are predicted as normal.', unsafe_allow_html = True)
    st.subheader('Model trained on the last dataset (*increased dataset*)')
    st.write('\n\n')
    st.write(
        '\n\n'
        '<u>Normalized confusion matrix</u>', unsafe_allow_html = True)
    cm_img = Image.open('static/VGG16_second_model_heatmap_confusion_matrix.jpg')
    st.image(cm_img, use_column_width = True)
    st.write(
        '\n\n'
        '<u>Classification report</u>', unsafe_allow_html = True)

    classification_secondmodel = pd.DataFrame({
        "precision":[0.96,0.93,0.91," "," ", 0.93, 0.93],
        "recall": [0.96, 0.93, 0.91," "," ", 0.93, 0.93],
        "f1-score": [0.96, 0.93, 0.91," ",0.93, 0.93, 0.93],
        "support": [229, 268,269, " ", 766, 766, 766]
    }, index=["COVID-19", "NORMAL", "VIRAL", " ", "accuracy", "macro avg", "weighted avg"])
    # Display classification report
    st.write(classification_secondmodel)

    st.write('\n\n')
    st.write(
        '\n\n'
        'We obtained an accurate classification on the test set with <strong>93% accuracy</strong>, <strong>99% AUC</strong> and <strong>93% F1-score</strong>.'
        '\n\n'
        'The <strong>recall</strong> for <em>viral pneumonia</em> is better with <strong>91%</strong> instead of 88%. Interestingly, the <strong>recalls</strong> for <em>COVID-19</em> and <em>normal</em> X-rays are lower with respectively <strong>96%</strong> instead of 98% regarding COVID-19 and <strong>93%</strong> instead of 96% regarding normal X-rays.', unsafe_allow_html = True)
    st.subheader('Fine-tuned model')
    st.write('\n\n')
    st.write(
        '\n\n'
        '<u>Normalized confusion matrix</u>', unsafe_allow_html = True)
    ft_cm_img = Image.open('static/VGG16_finetuned_second_model_heatmap_confusion_matrix.jpg')
    st.image(ft_cm_img, use_column_width = True)
    st.write(
        '\n\n'
        '<u>Classification report</u>', unsafe_allow_html = True)

    classification_secondmodel_tuned = pd.DataFrame({
        "precision":[0.99,0.96,0.97," "," ", 0.97, 0.97],
        "recall": [1.00, 0.97, 0.96," "," ", 0.98, 0.97],
        "f1-score": [1.00, 0.96, 0.97," ",0.97, 0.97, 0.97],
        "support": [229, 268,269, " ", 766, 766, 766]
    }, index=["COVID-19", "NORMAL", "VIRAL", " ", "accuracy", "macro avg", "weighted avg"])
    # Display classification report
    st.write(classification_secondmodel_tuned)

    st.write('\n\n')
    st.write(
        '\n\n'
        'We obtained our best classification result with this fine-tuned model on the test set with <strong>97% accuracy</strong>, <strong>100% AUC</strong> and <strong>98% F1-score</strong>.', unsafe_allow_html = True)

# #################
# Page Prediction #
# #################

def page_prediction(state):
    st.title("Chest Radiography Image Classification")
    st.write('\n\n')
    st.subheader("Upload a chest X-Ray for image classification as COVID-19, Normal or Viral Pneumonia")
    uploaded_file = st.file_uploader("Choose a chest X-Ray ...", type = ["jpg", "png"], key="1")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-Ray.', use_column_width=True)
        st.write("")
        image_loaded = st.text("Wait for classification...")
        label = import_and_predict(image)
        image_loaded.text("Classification done!")
        if label == 0:
            st.warning("The model predicts the patient has **COVID-19**.")
        elif label == 1:
            st.success("The model predicts the patient is **healthy**.")
        else:
            st.write("The model predicts the patient has **Viral Pneumonia**.")
        st.info('It can be very interesting to **visualize** directly what the **Convolutional Neural Network** values in the images for predicting the condition.'
            '\n\n'
            "For non-technical people in the medical field, especially in *radiology*, it is important to know how the model as discriminate healthy patient X-Rays from that of a COVID-19 case or Viral Pneumonia case."
            '\n\n'
            "Hence, the **GRADient-weighted Class Activation Mapping** (*Grad-CAM*) is a technic aiming at visualizing the features used for classification by generating a **heatmap** superimposed on the image."
            '\n\n'
            'Click below to produce a *Grad-CAM* visualization')
        if st.checkbox("Produce Grad-CAM"):
            if uploaded_file is not None:
                image2 = Image.open(uploaded_file)
                img_array = get_img_array(image2)
                heatmap = make_gradcam_heatmap(img_array)
                gradcam_image = display_grad_img(image2, heatmap)
                st.image(gradcam_image, use_column_width=True)
                st.success("By highlighting the most important features, we could potentially show the characteristics of contracting COVID-19 and those features could be used for further model exploration such as the evolution of the disease through time, or a comparison of pulmonaries symptoms between several diseases.")
                st.warning('We see from the localization map that characteristics of COVID-19 is often at the **endings of the lungs**, that is presumably the inflammation of the alveoli (i.e. the small air sacs containing oxygen that crosses into bloodstreams are filled up by fluid). Also, recent studies highlighted that, contrarily to other pneumonia-caused viruses, COVID-19 is more frequently affecting both lungs, but also that it is a multi-visceral disease that can affect also liver, kidneys, heart, nerves.'
                '\n\n'
                "It could be helpful to look at a more 'global' picture of human body, which might therefore highlight other body parts being stricken by the virus")
                

            
# #################
# Page Conclusion #
# #################

def page_conclusion(state):
    st.title('Summary')
    st.write('\n\n')
    st.write(
        "Working on this project at the very moment of the outbreak made us feel useful and helpful as we're trying to assist radiologists and all the medical personnel hardly working in the front line."
        '\n\n'
        "What's interesting is that we've showcased our knowledge and skills on two fields that's breathtaking to us and that weaves perfectly together: Artificial Intelligence and Medicine."
        '\n\n'
        "We had a real pleasure in working on this project and it has been an unforgettable journey along which we've struggled with figuring out how to deal with the imbalanced dataset regarding the COVID-19 images, building a multiclass classification model that's not overfitting or trying to work out the features contribution of our CNN."
        '\n\n'
        "Nonetheless, we're proud of displaying our work in a WebApp that hopefully will teach and entertain at the same time."
        '\n\n'
        'To summarize, with our model, we could demonstrate the efficiency of <strong>deep transfer learning</strong> for radiography analyses and diagnosis, with an appropriate architecture and fine-tuning parameters.', unsafe_allow_html = True)

# #####################################

if __name__ == '__main__':
    main()