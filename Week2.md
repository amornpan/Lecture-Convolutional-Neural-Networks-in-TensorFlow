Image augmentation and data augmentation is one of the most widely used tools in deep learning to increase your dataset size and make your neural networks perform better. In this week, you'll learn how the use of the easy-to-use tools in TensorFlow to implement this. Yes. So like last week when we looked at cats versus dogs, we had 25,000 images. That was a nice big dataset, but we don't always have access to that. In fact, sometimes, 25,000 images isn't enough. Exactly. Some of the nice things with being able to do image augmentation is that we can then, I think you just use the term create new data, which is effectively what we're doing. So for example, if we have a cat and our cats in our training dataset are always upright and their ears are like this, we may not spot a cat that's lying down. But with augmentation, being able to rotate the image, or being able to skew the image, or maybe some other transforms would be able to effectively generate that data to train off. So you skew the image and just toss that into the training set. But there's an important trick to how you do this in TensorFlow as well to not take an image, warp it, skew it, and then blow up the memory requirements. So TensorFlow makes it really easy to do this. Yes. So you will learn a lot about the image generator and the image data generator, where the idea is that you're not going to edit the images directly on the drive. As they get float off the directory, then the augmentation will take place in memory as they're being loaded into the neural network for training. So if you're dealing with a dataset and you want to experiment with different augmentations, you're not overriding the data. So [inaudible] to generate a library lets you load it into memory and just in memory, process the images and then stream that to the training set to the neural network we'll ultimately learn on. This is one of the most important tricks that the deep learning [inaudible] realizes, really the preferred way these days to do image augmentation. Yeah and I think it's, for the main reason that it's not impacting your data, right, you're not overriding your data because you may need to experiment with that data again and those kind of things. It's also nice and fast. It doesn't blow up your memory requirements. You can take one image and create a lot of other images from it, but you don't want to save all those other images onto this. Remember, we had a conversation recently about the lack of, there's a lot of literature on this topic so there's opportunity to learn. Yeah. One of these thinkings about data augmentation and image augmentation is so many people do it, it's such an important part of how we train neural networks. At least today, the academic literature on it is thinner relative to what one might guess, given this importance, but this is definitely one of the techniques you should learn. So please dive into this week's materials to learn about image augmentation and data augmentation.


You'll be looking a lot at Image Augmentation this week.

Image Augmentation is a very simple, but very powerful tool to help you avoid overfitting your data. The concept is very simple though: If you have limited data, then the chances of you having data to match potential future predictions is also limited, and logically, the less data you have, the less chance you have of getting accurate predictions for data that your model hasn't yet seen. To put it simply, if you are training a model to spot cats, and your model has never seen what a cat looks like when lying down, it might not recognize that in future.

Augmentation simply amends your images on-the-fly while training using transforms like rotation. So, it could 'simulate' an image of a cat lying down by rotating a 'standing' cat by 90 degrees. As such you get a cheap way of extending your dataset beyond what you have already.

To learn more about Augmentation, and the available transforms, check out https://github.com/keras-team/keras-preprocessing -- and note that it's referred to as preprocessing for a very powerful reason: that it doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset.


To this point, we've been creating convolutional neural networks that train to recognize images in binary classes. Horses or humans, cats or dogs. They've worked quite well despite having relatively small amounts of data to train on. But we're at a risk of falling into a trap of overconfidence caused by overfitting. Namely, when the dataset is small, we have relatively few examples and as a result, we can have some mistakes in our classification. You've probably heard us use the term overfitting a lot and it's important to understand what that is. Think of it as being very good at spotting something from a limited dataset, but getting confused when you see something that doesn't match your expectations. So for example, imagine that these are the only shoes you've ever seen in your life. Then, you learn that these are shoes and this is what shoes look like. So if I were to show you these, you would recognize them as shoes even if they are different sizes than what you would expect. But if I were to show you this, even though it's a shoe, you would likely not recognize it as such. In that scenario, you have overfit in your understanding of what a shoe looks like. You weren't flexible enough to see this high-heel as a shoe because all of your training and all of your experience in what shoes look like are these hiking boots. Now, this is a common problem in training classifiers, particularly when you have limited data. If you think about it, you would need an infinite dataset to build a perfect classifier, but that might take a little too long to train. So in this lesson, I want to look at some tools that are available to you to make your smaller datasets more effective. We'll start with a simple concept, augmentation. When using convolutional neural networks, we've been passing convolutions over an image in order to learn particular features. Maybe it's the pointy ears for cat, two legs instead of four for human, that kind of thing. Convolutions have been very good at spotting these if they're clear and distinct in the image. But if we could go further, what if for example we could transform the image of the cat so that it could match other pictures of cats where the ears are oriented differently? So if the network was never trained for an image of a cat reclining like this, it may not recognize it. If you don't have the data for a cat reclining, then you could end up in an overfitting situation. But if your images are fed into the training with augmentation such as a rotation, the feature might then be spotted, even if you don't have a cat reclining, your upright cat when rotated, could end up looking the same.

Ok, now that we've looked at Image Augmentation implementation in Keras, let's dig down into the code.

You can see more about the different APIs at the Keras site here: https://keras.io/preprocessing/image/

So if you remember the image generator class that we used earlier, it actually has the ability to do this for you. Indeed, you have already done a little image augmentation with it when you're re-scaled upon loading. That saved you from converting all of your images on the file system and then loading them in, you just re-scaled on the fly. So let's take a look at some of the other options. Here's how you could use a whole bunch of image augmentation options with the image generator adding onto re-scale. Rotation range is a range from 0-180 degrees with which to randomly rotate images. So in this case, the image will rotate by random amount between 0 and 40 degrees. Shifting, moves the image around inside its frame. Many pictures have the subject centered. So if we train based on those kind of images, we might over-fit for that scenario. These parameters specify, as a proportion of the image size, how much we should randomly move the subject around. So in this case, we might offset it by 20 percent vertically or horizontally. Shearing is also quite powerful. So for example, consider the image on the right. We know that it's a person. But in our training set, we don't have any images of a person in that orientation. However, we do have an image like this one, where the person is oriented similarly. So if we shear that person by skewing along the x-axis, we'll end up in a similar pose. That's what the shear_range parameter gives us. It will shear the image by random amounts up to the specified portion in the image. So in this case, it will shear up to 20 percent of the image. Zoom can also be very effective. For example, consider the image on the right. It's obviously a woman facing to the right. Our image on the left is from the humans or horses training set. It's very similar but it zoomed out to see the full person. If we zoom in on the training image, we could end up with a very similar image to the one on the right. Thus, if we zoom while training, we could spot more generalized examples like this one. So you zoom with code like this. The 0.2 is a relative portion of the image you will zoom in on. So in this case, zooms will be a random amount up to 20 percent of the size of the image. Another useful tool is horizontal flipping. So for example, if you consider the picture on the right, we might not be able to classify it correctly as our training data doesn't have the image of a woman with her left hand raised, it does have the image on the left, where the subjects right arm is raised. So if the image were flipped horizontally, then it becomes more structurally similar to the image on the right and we might not over-fit to right arm raisers. To turn on random horizontal flipping, you just say horizontal_flip equals true and the images will be flipped at random. Finally, we just specify the fill mode. This fills in any pixels that might have been lost by the operations. I'm just going to stick with nearest here, which uses neighbors of that pixel to try and keep uniformity. Check the carets documentation for some other options. So that's the concept of image augmentation. Let's now take a look at cats versus dogs that are trained with and without augmentation, so that we can see the impact that this has.

Now that you've gotten to learn some of the basics of Augmentation, let's look at it in action in the Cats v Dogs classifier.

First, we'll run Cats v Dogs without Augmentation, and explore how quickly it overfits.

If you want to run the notebook yourself, you can find it here: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb

Let's take a look at training cats versus dogs with a smaller dataset over a longer cycle. I'm going to start the training and we'll see it picked up the 2,000 training images, but note that we're training for 100 epochs. After the first epoch, we can see that our accuracy is 0.5345 and a validation accuracy is 0.5290. Keep an eye on those figures. Let's watch them for a few more epochs. We'll watch the accuracy and validation accuracy figures. After eight epochs, the accuracy is approaching 0.8, but the validation accuracy has slowed its growth. So now let's skip ahead to the end. I'm going to plot the accuracy and loss overall 100 epochs. We can see from this figure that the training reach close to a 100 percent accuracy in a little over 20 epochs. Meanwhile, the validation topped out at around 70 percent, and that's overfitting clearly been demonstrated. In other words, the neural network was terrific at finding a correlation between the images and labels of cats versus dogs for the 2,000 images that it was trained on, but once it tried to predict the images that it previously hadn't seen, it was about 70 percent accurate. It's a little bit like the example of the shoes we spoke about earlier. So in the next video, we'll take a look at the impact of adding augmentation to this.

Now that we’ve seen it overfitting, let’s next look at how, with a simple code modification, we can add Augmentation to the same Convolutional Neural Network to see how it gives us better training data that overfits less!

In the previous video, we looked at training a small data set of cats versus dogs, and saw how overfitting occurred relatively early on in the training, leading us to a false sense of security about how well the neural network could perform. Let's now take a look at the impact of adding image augmentation to the training. Here we have exactly the same code except that we've added the image augmentation code to it. I'll start the training, and we'll see that we have 2,000 training images in two classes. As we start training, we'll initially see that the accuracy is lower than with the non-augmented version we did earlier. This is because of the random effects of the different image processing that's being done. As it runs for a few more epochs, you'll see the accuracy slowly climbing. I'll skip forward to see the last few epochs, and by the time we reach the last one, our model's about 86 percent accurate on the training data, and about 81 percent on the test data. So let's plot this. We can see that the training and validation accuracy, and loss are actually in step with each other. This is a clear sign that we've solved the overfitting that we had earlier. While our accuracy is a little lower, it's also trending upwards so perhaps a more epochs will get us closer to 100 percent. Why don't you go ahead and give it a try?

Having clearly seen the impact that augmentation gives to Cats v Dogs, let’s now go back to the Horses v Humans dataset from Course 1, and take a look to see if the augmentation algorithms will help there! Here’s the notebook if you want to try it for yourself!

Of course, image augmentation isn't the magic bullet to cure overfitting. It really helps to have a massive diversity of images. So for example, if we look at the horses or humans data set and train it for the same epochs, then we can take a look at its behavior. So I'm going to start training and show all 100 epochs. I sped it up a bit to save your time. As you watch, you'll see the test accuracy climbing steadily. At first, the validation accuracy seems to be in step, but then you'll see it varying wildly. What's happening here is that despite the image augmentation, the diversity of images is still too sparse and the validation set may also be poorly designed, namely that the type of image in it is too close to the images in the training set. If you expect the data for yourself you'll see that's the case. For example, the humans are almost always standing up and in the center of the picture, in both the training and validation sets, so augmenting the image will change it to look like something that doesn't look like what's in the validation set. So by the time the training has completed, we can see the same pattern. The training accuracy is trending towards 100 percent, but the validation is fluctuating in the 60s and 70s. Let's plot this, we can see that the training accuracy climbs steadily in the way that we would want, but the validation fluctuated like crazy. So what we can learn from this is that the image augmentation introduces a random element to the training images but if the validation set doesn't have the same randomness, then its results can fluctuate like this. So bear in mind that you don't just need a broad set of images for training, you also need them for testing or the image augmentation won't help you very much.

# Augmentation: A technique to avoid overfitting

## 2.2.1 Image augmentation is tool to avoid overfitting

**Image augmentation** and **data augmentation** is one of the most widely used tools in deep learning to 
- increase your dataset size (useful when you have limited data) and 
- make your neural networks perform better (avoid overfitting). 

``` 
So for example, if we have a cat and our cats in our training dataset are always upright and their ears are like this, we may not spot发现 a cat that's lying down. 
```

But with **augmentation**, being able to **rotate** the image, or being able to **skew** the image, or maybe some other **transforms** would be able to effectively generate that data to train off.

|Original|Augmentation|New images|Better performance|
|-|-|-|-|
|![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img2/cat1.jpg width="100")| (crap)--> |<img src="./img2/cat2.jpg" width=100/>|Better performance|
||(rotate)-->|<img src="./img2/cat3.jpg" width=100/>||
||

**Issue**:
- Don't do image augmentation on drive directly, which will overwrite the data.
- It's trick to do it since it can brow up memories easily. 

**Sol**:
- Tensorflow will handle this using Image generator and Image data generator (e.g. [ImageDataGenerator](https://keras.io/preprocessing/image/)).
- Augmentation takes place in memory **as** images are being loaded into the neural network for training (**amends your images on-the-fly**). Then stream them into the training set to the neural network we are learning on. All happens in memory.


## 2.2.2 Introducing augmentation

**Reason** of overfitting: 
- limited images.

## 2.2.3 ImageDataGenerator() class

```python
train_datagen = ImageDataGenerator(
    rescale = 1./255 # re-scale on the fly
    rotation_range = 40, # Rotation range is a range from 0-180 degrees with which to randomly rotate images. 
    width_shift_range = 0.2,# Shifting, moves the image around inside its frame.
    height_shift_range = 0.2,
    shear_range = 0.2, # shear the image by random amounts up to the specified portion (up to 20% in this case)
    zoom_range = 0.2, # zoom in a random amount(percent) up to 20% of the size of the image
    horizontal_flip = True,
    fill_mode = 'nearest') # fills in any pixels that might have been lost by the operations. I'm just going to stick with nearest here, which uses neighbors of that pixel to try and keep uniformity. 
```

- **shift**:
  
  Shifting, moves the image around inside its frame. Many pictures have the subject centered. So if we train based on those kind of images, we might over-fit for that scenario. These parameters specify, as a proportion of the image size, how much we should randomly move the subject around. So in this case, we might offset it by 20% vertically or horizontally. 

- **shear**:

|Can't recognize|What we have|Shear that person by skewing歪斜 along the x-axis we'll end up in a similiar pose|
|-|-|-|
|<img src="./img2/shear1.png" width=200/>|<img src="./img2/shear2.png" width=200/>|<img src="./img2/shear3.png" width=200/>|
||

- **zoom**:
  
|Can't recognize|What we have|Zoom in that training image|
|-|-|-|
|<img src="./img2/zoom1.png" width=200/>|<img src="./img2/zoom2.png" width=200/>|<img src="./img2/zoom3.png" width=200/>|
||

- **horizontal flip**:
  
|Can't recognize|What we have|If the image were flipped horizontally, then it becomes more structurally similar to the image in test data and we might not over-fit to right arm raisers.|
|-|-|-|
|<img src="./img2/flip1.png" width=200/>|<img src="./img2/flip2.png" width=200/>|<img src="./img2/flip3.png" width=200/>|
||

- **fill_mode**: is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.



**Q**: When training with augmentation, you noticed that the training is a little slower. Why?

**A:**
- [ ] Because the augmented data is bigger

- [ ] Because there is more data to train on

- [x] Because the image processing takes cycles

- [ ] Because the training is making more mistakes

------

## 2.2.4 Without Augmentation V.S. With Augmentation

Cats-vs-Dogs without & with Augmentation: [official code](./myExercise/Cats_v_Dogs_Augmentation.ipynb), [official link](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb#scrollTo=gGxCD4mGHHjG),

[My code](./myExercise/Copy_of_Cats_v_Dogs_Augmentation.ipynb)


We have 4 convolutional layers with 32, 64, 128 and 128 convolutions respectively. 2000 images for training and 100 images for testing. We ran 100 epoches。

In terms of code, the **only difference** is:

- **Without Augmentation** (Left image below):

```python
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
```

- **With Augmentation** (Right image below):

```python
# This code has changed. Now instead of the ImageGenerator just rescaling
# the image, we also rotate and do other operations
# Updated to do image augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
```

<img src="./img2/catsdogswithout.png" width=300/><img src="./img2/catsdogswith.png" width=300/>

**Left**: We can see from this figure that the **training** reach close to a 100 percent accuracy in a little over 20 epochs. Meanwhile, the **validation** topped out at around **70** percent, and that's **overfitting** clearly been demonstrated. In other words, the neural network was terrific极好的 at finding a correlation between the images and labels of cats versus dogs for the 2,000 images that it was trained on, but once it tried to predict the images that it previously hadn't seen, it was about 70 percent accurate.

**Right**:
- Initial phase: We'll initially see that the accuracy is lower than with the non-augmented version we did earlier. This is because of the random effects of the different image processing that's being done.
- Solved: We can see that the training and validation accuracy, and loss are actually in step with each other. This is a clear sign that we've solved the overfitting that we had earlier.

------

## 2.2.5 Tri it yourself (**Problem**!): Horse-or-Human-WithAugmentation

[official code](./myExercise/Horse_or_Human_WithAugmentation.ipynb),
[official link](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb)

[My code](./myExercise/Copy_of_Horse_or_Human_WithAugmentation.ipynb)

`Short summary: A broad set of images are needed for bothen training and testing. `

<img src="./img2/horseshuman.png" width=400/>

Of course, image augmentation isn't the magic bullet to cure overfitting. It really helps to have a massive大量的 diversity of images. 

As you watch, you'll see the test accuracy climbing steadily. At first, the **validation accuracy** seems to be in step, but then you'll see it **varying wildly**. 

What's happening here is that despite the image augmentation, the diversity of images is still too sparse and the validation set may also be poorly designed, namely that the type of image in it is too close to the images in the training set.

```
For example, the humans are almost always standing up and in the center of the picture, in both the training and validation sets, so augmenting the image will change it to look like something that doesn't look like what's in the validation set. 
```

So what we can learn from this is that the **image augmentation** introduces引入 a **random** element to the training images but if the **validation set** doesn't have the same **randomness**, then its results can fluctuate like this. 

So bear in mind that you **don't just** need a broad set of images for **training**, you also need them for **testing** ***OR*** the image augmentation won't help you very much.

------

## 2.2.6 Exercise 6: Cats v Dogs full Kaggle Challenge exercis

[Question](./myExercise/Exercise_6_Question.ipynb)

[official sol code](./myExercise/Copy_of_Exercise_6_Answer.ipynb), [official link](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%206%20-%20Cats%20v%20Dogs%20with%20Augmentation/Exercise%206%20-%20Answer.ipynb)

[My Sol code](./)

<img src="./img2/exercise6.png" width=400/>
