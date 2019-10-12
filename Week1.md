In the first course, you learned how to use TensorFlow to implement a basic neural network, going up all the way to basic Convolutional Neural Network. In this second course, you go much further. In the first week, you take the ideas you've learned, and apply them to a much bigger dataset of cats versus dogs on Kaggle. Yes so we take the full Kaggle dataset of 25,000 cats versus dogs images. In the last module, we looked at horses and humans, which was about 1,000 images. So we want to take a look at what it's like to train a much larger dataset, and that was like a data science challenge, not that long ago. Now, we're going to be learning that here, which I think is really cool In fact, we have substantially similar ideas as their previous goals, and apply it to much bigger datasets, and hopefully get great results. Yeah, we're hoping for good results. Let's see what the students get as they do some of the assignments with it as well. One of the things that working with a larger dataset, then helps with is over-fitting. So with a smaller dataset, you are at great risk of overfitting; with a larger dataset, then you have less risk of over-fitting, but overfitting can still happen. Pretty cool. Then in week 2, you'll learn another method for dealing with overfitting, which is that TensorFlow provides very easy to use tools for data augmentation, where you can, for example, take a picture of a cat, and if you take the mirror image of the picture of a cat, it still looks like a cat. So why not do that, and throw that into the training set. Exactly. Or for example, you might only have upright pictures of cats, but if the cat's lying down, or it's on its side, then one of the things you can do is rotate the image. So It's like part of the image augmentation, is rotation, skewing, flipping, moving it around the frame, those kind of things. One of the things I find really neat about it, is particularly if you're using a large public dataset, is then you flow all the images off directly, and the augmentation happens as it's flowing. So you're not editing the images themselves directly. You're not changing the dataset. It all just happens in memory. This is all done as part of TensorFlow's Image Generation [inaudible]? Exactly. That they'll learned about it in the second week. Yeah. So then too one of the other strategy, of course for avoiding overfitting, is to use existing models, and to have transfer learning. Yeah. So I don't think anyone has as much data as they wish, for the problems we really care about. So Transfer Learning, lets you download the neural network, that maybe someone else has trained on a million images, or even more than a million images. So take an inception network, that someone else has trained, download those parameters, and use that to bootstrap your own learning process, maybe with a smaller dataset. Exactly. That has been able to spot features that you may not have been able to spot in your dataset, so why not be able to take advantage of that and speed-up training yours. I particularly find that one interesting as you move forward. That to be able to build off of other people's work, the open nature of the AI community, I find is really exciting and that allows you to really take advantage of that and be a part of the community. Standing on the shoulders of giants. I use transfer learning all the time, so TensorFlow lets you do that easily [inaudible] open source. Then finally in the fourth week, Multicast learning. Rather than doing two classes, like horses verses humans, or cats verses dogs, what if you have more than two classes, like class five rock, paper, scissors, that would be three classes, or inception would be 1,000 classes. So that the techniques of moving from two to more than two, be it three or be it a 1,000, are very very similar. So we're going to look at those techniques and we'll look at the code for that. So and we have a rock, paper, scissors example, that you're going to be able to build off of. So in this second course, you take what you learned in the first course, but go much deeper. One last fun thing, Lawrence had seen this coffee mug into AI for everyone in the course, and he asked me to bring it. I love that course, so thank you so much. It's a great course, because it has got everything for people who are beginning; even people who are non-technical, all the way up to experts. So thank you for the mug, but is it okay if I say I spot a sports car in the course as well, would you bring that? I don't have one of those to bring to you. So I'm really excited about this course. Please go ahead and dive into the first of the materials for week 1.



We've gotten a lot of questions about whether this course teaches TensorFlow 2.0 or 1.x.

The answer is: Both!

We've designed the curriculum for the early modules in this course to have code that's generic enough that it works with both versions, so you are welcome to try it with the 2.0 alpha. If you are using the colabs, you'll use the latest version of TensorFlow that Google Colaboratory supports, which, at time of writing is 1.13. You can replace this with 2.0 by adding a codeblock containing:

!pip install tensorflow==2.0.0-alpha0 

...and you should be able to use TF2.0 alpha if you like!

Later modules may use specific versions of libraries, some of which may require TF2.0, and we will note them when you get there!


What does it take to download a public dataset off the Internet, like cats verses dogs, and get a neural network to work on it? Data is messy, sometimes you find surprising things like pictures of people holding cats or multiple cats or surprising things in data. In this week, you get to practice with using TensorFlow to deal with all of these issues. Yeah, and it's like, so even for example, you might have some files that are zero length and they could be corrupt as a results. So it's like using your Python skills, using your TensorFlow skills to be able to filter them out. Building a convolutional net to be able to spot things like you mentioned, a person holding it up. So that's some of the things we'll do this week, is by using, and it's still a very clean dataset that we're using with cats versus dogs, but you're going to hit some of those issues. I think you'll learn the skills that you need to be able to deal with other datasets that may not be as clean as this one. Yeah. Sometimes people think that AI is people like Lawrence and me sitting in front of a white board maybe a zen garden outside, talking about the future of humanity. The reality is, there's a lot of data cleaning, and having great tools to help with that data cleaning makes our workflow much more efficient. Definitely. So in this week, you get to practice all that, as well as train a pretty cool neural network to classify cats versus dogs. Please dive in.


In the next video, you'll look at the famous Kaggle Dogs v Cats dataset: https://www.kaggle.com/c/dogs-vs-cats

This was originally a challenge in building a classifier aimed at the world's best Machine Learning and AI Practitioners, but the technology has advanced so quickly, you'll see how you can do it in just a few minutes with some simple Convolutional Neural Network programming.

It's also a nice exercise in looking at a larger dataset, downloading and preparing it for training, as well as handling some preprocessing of data. Even data like this which has been carefully curated for you can have errors -- as you'll notice with some corrupt images!

Also, you may notice some warnings about missing or corrupt EXIF data as the images are being loaded into the model for training. Don't worry about this -- it won't impact your model! :)
https://www.kaggle.com/c/dogs-vs-cats

We've gone from the fashion dataset where the images were small and focused on the subject, to a new situation where we had images of horses and humans and action poses. We use convolutions to help us identify features in the image regardless of their location. This is a nice primer in solving some common data science problems on places like Kaggle. We'll next look at an old competition where you were encouraged to build a classifier to determine cats versus dogs. If you're not familiar with Kaggle, it's where ML challenges are posted often with prizes. Cats versus dogs was a famous one from a few years back. The techniques you've just learned can actually apply to that problem. So let's recap some of the concepts. One of the nice things with TensorFlow and Keras is that if you put your images into named subdirectories, an image generated will auto label them for you. So the cats and dogs dataset you could actually do that and you've already got a massive head start in building the classifier. Then you can subdivide that into a training set and a validation set. Then you can use image generators that appointed at those folders. To use an image generator, you should create an instance of one. If the data isn't already normalized, you can do that with the rescale parameter. You then call the flow from directory to get a generator object. For the training dataset, you will then point at the training directory and then specify the target size. In this case, the images are an all shapes and sizes. So we will resize them to 150 by 150 on the fly. We'll set the batch sizes to be 20. There's 2,000 images, so we'll use a 100 batches of 20 each. Because there are two classes that we want to classify for its still stays as a binary class mode. Similarly for validation, we set up a generator and pointed at the validation directory. We can explore the convolutions and pooling and the journey of the image through them. It's very similar to what you saw with the horses and humans. It has three sets of convolutions followed by pooling. Of course, the image is 150 by 150. Similarly, there's a single neuron with a sigmoid activation on the output. The summary of the layers is very similar to before but note that the size changes. We start with 150 by 150. So the convolution reduces that to 148 by 148. From there, we'll go until we end up with 17 by 17 that we feed into the dense layers. Compilation is as before. Now remember you can tweak the learning rate by adjusting the lr parameter. So now to train, and we can call model.fit generator and pass it the training generator and the validation generator. That's it. As you can see, it's very similar to what you built for horses versus humans. So let's see it in action.

Now that we’ve discussed what it’s like to extend to real-world data using the Cats v Dogs dataset from an old Kaggle Data Science challenge, let’s go into a notebook that shows how to do the challenge for yourself! In the next video, you’ll see a screencast of that notebook in action. You’ll then be able to try it for yourself.
https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb

In the last video you saw a screencast of the notebook that you can use as an example of how to build a classifier for Cats vs Dogs. You saw how, in some cases it didn’t classify one cat correctly, and we asked you to try to figure out how you might fix it for yourself. In the next video, you’ll see one solution to this.

Did you find a solution? Well, of course yours might be different from mine but let me show you what I did in the case of the cat my model thought was a dog. So let's go back to the notebook, and we'll run the code. I'll upload this image to see how it classifies. It's a crop of the cats, and lo and behold, it classifies as a cat. Let's open it, and compare it to the original image, and we'll see that just by cropping I was able to get it to change its classification. There must have been something in the uncropped image that matched features with a dog. Now I thought that was a very interesting experiment, didn't you? Now what do you think the impact of cropping might've had on training? Would that have trained the model to show that this was a cat better than an uncropped image. That's food for thought, and something to explore in the next lesson but first let's go back to the workbook.

Okay, in the previous video you took a look at a notebook that trained a convolutional neural network that classified cats versus dogs. Now let's take a look at how that worked.
0:10
Let's return to the notebook and take a look at the code that plots the outputs of the convolutions in max pulling layers. The key to this is understanding the model.layers API, which allows you to find the outputs and iterate through them, creating a visualization model for each one.
0:28
We can then load a random image into an array and pass it to the predict method of the visualization model.
0:35
The variable to keep an eye on is display_grid which can be constructed from x which is read as a feature map and processed a little for visibility in the central loop.
0:45
We'll then render each of the convolutions of the image, plus their pooling, then another convolution, etc.
0:52
You can see images such as the dog's nose being highlighted in many of the images on the left.
0:59
We can then run it again to get another random image. And while at first glance this appears to be a frog, if you look closely it's a Siamese cat with a dark head and dark paws towards the right of the frame. It's hard to see if any of the convolutions lock down on a feature. Except maybe the synonymous upright tail of the cat, we can see that vertical dark line in a number of the convolutions.
1:25
And let's give it one more try. We can see what's clearly a dog, and how the ears of the dog are represented very strongly. Features like this moving through the convolutions and being labeled as doglike could end up being called something like a floppy ear detector.


# Larger Dataset cats & dogs

## 2.1.1 ConvNet model

[Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats)

Data not clean:
- Data is messy, sometimes you find surprising things like pictures of people holding cats or multiple cats or surprising things in data.
- some files that are zero length and they could be corrupt as a results.
  


[Notebook](./myExercise/Course_2_Part_4_Lesson_2_Notebook.ipynb) will do:
- Explore the Example Data of Cats and Dogs (only 2000 of 25000 images)
- Build and Train a Neural Network to recognize the difference between the two
- Evaluate the Training and Validation accuracy

**Normalization**:

Preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range).
```python
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
```

**flow_from_directory()**:

Images come in all shapes and sizes. Before training a Neural network with them you'll need to **tweak** the images. Need them to be in a **uniform** size.

This ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .**flow_from_directory**(directory). 
Our generators will **yield** batches of 20 images of size 150x150 and their labels (binary).

```python
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))  
```

**Model**:

```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
```

**Optimizer**:

NOTE: In this case, using the **RMSprop** optimization algorithm is **preferable** to stochastic gradient descent (**SGD**), **because RMSprop automates learning-rate tuning for us**. (Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, and would work equally well here.)

**Training**:

```python
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_steps=50,
                              verbose=2)
```


## 2.1.2 Code
[Official code](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)

[My code](./myExercise/Course_2_Part_4_Lesson_2_Notebook.ipynb)



## 2.1.3 Fixing through cropping

In some cases it didn’t classify one cat correctly. Here is one solution to this.

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img2/kitty.jpg)
Calssified as cat.

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img2/cat1.jpg)
Calssified as dog.


![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img2/cat2.jpg)
Calssified as cat.


In the case of the cat my model thought was a dog. We'll see that just by **cropping** I was able to get it to change its classification. There must have been something in the uncropped image that matched features with a dog.

## 2.1.4 Validation not improve after 2 epochs

```python
history = model.fit_generator()
```
So we now have a **history** object that we can query for data. Call its history property passing at ACC which gets me the model accuracy. 

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img2/trainingandvalidationaccuracyperepoch.png)
![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img2/trainingandvalidationLOSSperepoch.png)

When I run it and plot the training and validation accuracy, we can see that my training went towards one while my validation leveled out into 0.7 to 0.75 range. That shows that my model isn't bad, but I **didn't** really **earn anything after just 2 epochs.** It fits the training data very well with the validation data needed work. These results are borne out in the loss where we can see that after two epochs, my training loss went down nicely, but my validation loss climbed. So as it is, my model is about 75 percent accurate-ish after two epochs, and I don't really need to train any further. Remember also that we just used a subset of the full data. Using the entire dataset would likely yield better results.

------

## 2.1.5 Exercise 5

Use the full Cats v Dogs dataset of 25k images. Note again that when loading the images, you might get warnings about EXIF data being missing or corrupt. Don't worry about this -- it is missing data in the images, but it's not visual data that will impact the training.

[Official code](./myExercise/Exercise_5_Answer.ipynb)

[My code](./myExercise/Exercise_5_Question.ipynb)






