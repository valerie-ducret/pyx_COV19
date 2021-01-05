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
    st.sidebar.info(
        'DataScientest Project - October 2020 Bootcamp Class'
        '\n\n'
        'Authors:'
        '\n\n'
        'Kévin PAME'
        '\n\n'
        'Valérie DUCRET')

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
        explode = (0.1, 0, 0)  # only "explode" the 1er slice (i.e. 'covid')
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

    st.header("Visualizations of X-rays")

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
        
        st.info("It is clearly difficult, particularly for untrained people, to detect the presence of COVID-19. The difficulty also comes with the low quality of some X-rays")
    
    st.header("See how works image generator from Keras")

    if st.button("Try image generator on COVID X-ray"):
        aug = ImageDataGenerator(rescale = 1./255,
                            rotation_range = 20,
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            zoom_range = 0.2,
                            shear_range = 0.2,
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
    st.warning('Given that two different datasets were used since the last update on *Kaggle*, two *identical* models were trained on the images. Here we will show you the results of both models and compare them as far as metrics are concerned.')
    st.write(
        'To build our efficient <strong>Convolutional Neural Network</strong>, we used the <strong>transfer learning</strong> technique which allows us to benefit from a relatively small computation time which aims to saving training time and weights of a <em>pre-trained</em> model to increase the algorithm performance. Finally, this technique does not need a huge amount of data to get really efficient prediction of COVID-19.'
        '\n\n'
        'Thus, we used a pre-trained <strong>VGG16</strong> model and built a relatively simple classification model.', unsafe_allow_html = True)
    st.header('Architecture')
    st.write('\n\n')
    img = Image.open('static/VGG16_architecture.jpg')
    st.image(img, use_column_width = True)
    st.write(
        '\n\n'
        'Interestingly, the simple addition of dense layers with decreasing units after the convolution part was clearly increasing classification accuracy.')
    
    st.header('Visualize the feature maps during convolution')
    st.write('\n\n')
    st.write(
        'Convolutional Neural Network models have impressive <strong>classification performance</strong>. Yet it is not clear <em>why</em> they perform so well and thus <em>how</em> they might be improved.'
        '\n\n'
        'Those models use linear filters, which results in activation maps or features maps. Both filters and feature maps can be visualized.'
        '\n\n'
        'For instance, we can try to understand small filters, such as <em>contour or line detectors</em>. Using feature maps that results from the filtering, we may even get insight into the <strong>internal representation</strong> that the model has of a particular input.', unsafe_allow_html = True)    
    st.write("")
    if st.button("Generate feature maps"):
        category_selected = st.text("Please wait...")
        path_covid="dataset/COVID"
        files_covid=os.listdir(path_covid)
        img_covid=random.choice(files_covid)
        image = path_covid+'/'+img_covid
        #n_layer = st.slider("output layer's number", 1,2)
        select_and_features_map(image) #, n_layer)
        category_selected.text("")
        st.info(
            'You are visualizing 64 feature maps as subplots. Those maps was generated from the first output of the convolution on a COVID-19 X-ray (dimensions 224x224x64).'
            '\n\n'
            'It is of course possible to look further into the model, to get a better idea of the most important features used for classification.')
            #'\n\n'
            #'You can now use the slider to move to the intermediate layers.')
        
    st.header('Training hyperparameters')
    st.write('\n\n')
    st.write(
        'The model was compiled with the <strong>Adam optimizer</strong> and an initial <strong>learning rate</strong> of <em>0.001</em>.'
        '\n\n'
        'The <strong>loss function</strong> used was a <strong>sparse categorical cross-entropy</strong> and the <strong>metrics</strong> used were <strong>accuracy</strong>, <strong>AUC</strong> (<strong><em>Area Under the Receiver Operating Characteristic Curve</em></strong>) and <strong>F1-score</strong>.'
        '\n\n'
        'The model was fit using the adjusted class weights on <em>20 epochs</em>. '
        '\n\n'
        'We then proceeded to fine-tuning the model by removing the last 10 layers of the pre-trained VGG16 model with an initial learning rate of <em>0.0001</em>, adding a “<strong>ReduceLROnPlateau</strong>” callback which aims for changing the learning rate with respect to a chosen metric on the test set.'
        '\n\n'
        'Here we chose to monitor the <strong>validation loss function</strong>, that is to say that the learning rate would be reduced when the quantity has stopped decreasing.'
        '\n\n'
        'The fine-tuned model was fit using the adjusted class weights on <em>30 epochs</em> and the aforementioned callback.', unsafe_allow_html = True)
        
# ###############################
# Page Results & Interpretation #
# ###############################

def page_results(state):
    st.title('Results')
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
    uploaded_file = st.file_uploader("Choose a chest X-Ray ...", type = ["jpg", "png"])
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
            
# #################
# Page Conclusion #
# #################

def page_conclusion(state):
    st.title('Summary')
    st.write('\n\n')
    st.write(
        "Working on this project at the very moment of the outbreak made us feel useful and helpful as we're trying to assist radiologists and all the medical personnel hardly working on the front line."
        '\n\n'
        "What's interesting is that we've showcased our knowledge and skills on two fields that's breathtaking to us and that weaves perfectly together: Artificial Intelligence and Medicine."
        '\n\n'
        "We had a real pleasure in working on this project and it has been an unforgettable journey along which we've struggled with figuring out how to deal with the imbalanced dataset regarding the COVID-19 images, building a multiclass classification model that's not overfitting or trying to work out the features contribution of our CNN."
        '\n\n'
        "Nonetheless, we're proud of displaying our work in a WebApp that hopefully will teach and entertain at the same time."
        '\n\n'
        'To summarize, with our model, we could demonstrate the efficiency of <strong>deep transfer learning</strong> for radiography analyses and diagnosis, with an appropriate architecture and fine-tuning parameters.', unsafe_allow_html = True)
    with st.beta_expander('Bonus'):
        st.write(
            'It can be very useful and interesting to <strong>visualize</strong> what the <strong>Convolutional Neural Network</strong> values when it does a prediction.'
            '\n\n'
            "For non-technical people in the medical field, especially in <em>radiology</em>, it's important to know the difference and discriminate <strong>healthy patient</strong> X-Rays from that of a <strong>COVID-19</strong> case or <strong>Viral Pneumonia</strong> case. Fortunately, it is possible to visualize what the CNN deems as <strong>significant features</strong>."
            '\n\n'
            'Hence, introducing the <strong>Grad-CAM class activation maps</strong> (<em>GRADient-weighted Class Activation Mapping</em>). Grad-CAM class activation maps generate <strong>heatmaps</strong> at the <em>convolutional</em> level rather than the <em>dense neural layer</em> level, taking into account more spatial details.'
            '\n\n'
            'Below, a localization map is produced highlighting the important regions in the image for predicting for COVID-19.', unsafe_allow_html = True)
        covid_gradcam = Image.open('static/gradcam_covid.jpg')
        st.image(covid_gradcam, use_column_width = True)
        st.info(
            'We see from the localization map that both lungs express the characteristics of COVID-19, particularly at the **end of the rib cage**, that is presumably the inflammation of the alveoli (i.e. the small air sacs containing oxygen that crosses into bloodstreams are filled up by fluid). Indeed, recent studies highlighted that, contrarily to other pneumonia-caused viruses, COVID-19 is affecting both lungs.')
        st.write(
            "For some unknown reasons, this interpretation technique didn't work properly for all images. Unfortunately, we didn't have enough time to focus more on figuring out how we could adjust the code behind Grad-CAM and display a set of images chosen by the user.", unsafe_allow_html = True)
        st.info(
            'There are improvements that can still be made to this project. One remaining problem is to understand how the model learned to classify the images for each category (model interpretability). By highlighting the most important features, we could potentially show the characteristics of contracting COVID-19 and those features could be used for further model exploration such as the evolution of the disease through time, or a comparison of pulmonaries symptoms between population categories (age, gender, ...).')

# #####################################

if __name__ == '__main__':
    main()