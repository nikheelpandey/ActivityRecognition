# HAR

Human activity recognition can be done using many ways. This repo present one of the easiest ways to do it.

- Get the keypoints from a keypoint detector 
- Train an LSTM model on the sequence of keypoints




## Data Preprocessing

We can extract the keypoints of each frame in every video using a keypoint detector and save the inferences as a sequence in a json file. Since, each video can have a variable length, the sequence length could be different for each video. The sequence length also depepnds upon the number of frames in which a subject it present over which the keypoint detection can be preformed. 

I conducted my experiments with 2 keypoint generators.

- Mediapipe  : Provides 33 keypoints (`keypoint_extraction_using_mpipe.ipynb`)
- Torchvision: Provides 17 keypoints ( `keypoint_extraction_using_heavier_model.ipynb`)

From the looks of it, it seems like since mediapipe is generating more number of keypoints, it would be a better choice. But I ran into an issue where mediapipe keypoint generation kept on failing on the validation videos. However, the torchvision library that generates only 17 points, had a similar performance on train and test set. 



#### Asumption
My solution primarily hinges on two assumptions:

- Since 1 second of video typically contains 24 to 30 frames, there is not a lot of variance from frame to frame in the keypoint locations. Hence, a fair assumtion that I made to reduce the inferencing cost and time is that instead of all the frames, every 5th frame has been used to get the keypoint locations.

- Sequences with fairly low length can be ignored as they won't have enough context. The hyperparameter: `min_det_threshold` deals with the number the said length.




# Data Overview


The data has to be loaded from the json files. The json files were created by looping over all the frames in a video and generating the keypoint location from the detector. 

### Data Loading

- Read the Jsons
- Keep the data of the json which have sequence length > `min_det_threshold`


### Refer Solution.ipynb for model training and performance analysis



## Sequence Modeling 

I experimented with various LSTM models for sequence classification. The best performance was observed in the ConvLSTM models. ConvLSTM. The Convolutional LSTM architectures bring together time series processing and computer vision by introducing a convolutional recurrent cell in a LSTM layer. 

Input shape: 5D tensor with shape - (samples, timesteps, filters, new_rows, new_cols)


### Model Hyperparameters

- max_seq_length: The maximum length of sequence while padding and trimming
- n_steps : timestep paramter
- n_length: length of sequences
- n_steps * n_length = max_seq_length 
- test_size : Ratio of train data that the model should use to evaluate
- num_conv_filter: Number of filters in convlstm layer
- filter_size: The size of the filters



# TODO


##### A better oversampling technique 

We can oversample the data multiple times and train a n-fold crossvalidation model on each of them. We can performe some analysis on the model performances on the individual oversampled-dataset. The deviation in the performance will tell us if there are anomalies. If the results vary too much across the oversampled data, then we have some anomalies/outliers that we need to take care of. If the performance is fairly same despite oversampling 20 times, then the dataset is fine.


##### Data Augmentation

Since there needs to be an increase in the data in order for the model to learn better representation, we need to apply some data augmentation strategy. Following are few augmentation strategies I could think of considering contrastive learning as the end goal  :

- Normallization 
    We can use the location of the centroid and normalize the keypoints to make them traslation invarient. 


- Reversing the squences:
    Some motions such as squat can easily be reversed and still they would make same sense if we follow the sequence through.

- Morphological Transformation:
    We can create more data by offsetting the location using small gaussian noise addition. Basically, 
    
        for frame in a sequence:
            for cordinate in the frame:
               cordintate (+-)= c
               
     Here c is a small deviation in the location
               
- Partial Sequences:
    We can partition the sequences into mini sequences of sufficient lengths. For an instance, a sequence can be sliced from either ends and can be padded to create multiple augmented sequences.  
    
    
During augmentation, we can combine these independent techniques to create a richer sample. 
    
    
               
               
##### Contrastive Learning 

Once the augmentation pipeline has been constructed, we go ahead and implement a SimCLR based framework for contrastive learning.

Contrastive learning is an approach to learn similarity of representation from data points that are organized in pairs of similar/dissimilar examples. The idea is to train ML models for the classification of similar and dissimilar sequence pairs. There are 3 main components to contrastive learning:

- Dataset - Samples of similar and dissimilar pairs. To be similar, a sequence can be augmented. The contrastive loss basically try to cluster togther the embeddings of similar samples and simultenously maximize the intra-cluser distances of different classes. 

- Representation - A mechanism to get a good representations from sequences for comparison. We can train an LSTM model on similar dataset and same input. Then we can remove the classification head thus resulting in an embedding generator for the video dataset.

- Similarity Metric - Measurement of similarity or distinctiveness. Usually, cosine smilarity is used to compare similarity of the embeddings.


