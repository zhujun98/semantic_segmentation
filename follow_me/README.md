# Semantic Segmentation - Follow Me

Jun Zhu

---

We will build a segmentation network to track VIP in an image. We use the data set for one of the projects at RoboND, Udacity. However, we have implemented different models. One is the original [SegNet](https://arxiv.org/pdf/1511.00561.pdf) and the other has a similar structure but all the convolutional layers are replaced by [depthwise separable convolutional layers](https://arxiv.org/pdf/1610.02357.pdf).


## Models

Due to technical constraint, **we have not combined the pooling indices in the encoder to the decoder!!!**

## Results

TODO: increase the model size

Both models were trained on AWS EC2 g3.4xlarge instance.

| Model                                  | SegNet           | Depthwise SegNet  |
| -------------------------------------------|:---------------------:| -----------------------------:|
| No. epochs                          |                 |                         |
| No. parameters (M)              |                |                         |
| Training time per epoch (s)  |                |                        |
| Training loss                        |        |                     |
| Validation loss                     |          |                         |