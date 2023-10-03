# Mini Project 1 - CMSC818B Decisioin Making for Robotics

## ICRA 2023 Workshop - ScalableAD
_This workshop presents good literature related to solving challenging problems in Autonomous Driving Systems_
<br>
[ScalableAD website](https://sites.google.com/view/icra2023av)

Demo equation: 
<br/>
![formula](https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1)
<br/>
![formula](https://render.githubusercontent.com/render/math?math= \sqrt{3x-1}+(1+x)^2)

**Topic 1: F2BEV: Bird's Eye View Generation from Surround-View Fisheye Camera Images for Automated Driving.**

**Topic 2: Scaling AV through Adversarial Scenario Generation and Continual Learning.**
<br/>
_Talk by: Yuning Chai (Head of AI Research at Cruise)_

Dr. Chai talks about how generating adversarial scenarios using GANs can help in providing additional learning examples for the agent. He addresses how generative AI can be used to provide a scalable technology that can generate good training data. This is essential to help in expanding the categories of scenarios that are available as training data. 

Modern neural networks are composed of billions of parameters that are very data hungry and need large amounts of data (often in terabytes or even petabytes) to generalize well. The talk  proposes a method of multi-pass learning in which we can eliminate the need of storing old data without sacrificing the performance metrics on old data. This helps in solving the scalability issue in continual learning in AI models.

Generative AI paves the way to generate realistic examples in high-dimensional spaces such as RGB images or 3-dimensional paths. The talk makes an important contribution of using generative AI to generate realistic but improbable pedestrian behavior. This has been accomplished by modifying the architecture of the generator of a traditional GAN where a single neural network is used to produce the output. Here, an encoder and a decoder are used to make up the generator network. An additional embedding is generated from the desired distribution and passed to the decoder that generates an output sample (as seen in fig. 1). By using this method, the authors improve the chances of obtaining ‘valid’ samples that will be more suitable candidates for training the underlying neural network.

<p align="center">
  <img src="images/fig1_cruise.png" alt="Encoder Decoder GAN" width="600" />
  <em>Fig 1: Modified GAN architecture</em>
</p>

Another important contribution in the area of continual learning of neural networks has been made. In most approaches, access to old data (not for retraining, but for evaluation) is often required or the approach is not scalable to multiple fine-tuning stages. In the proposed method, the weight updates are made based on only the new data but using the original checkpoint model’s weights as reference. An example can be seen in fig. 2, where the weight updates are calculated by a combination of gradients obtained from the current model and new data and the difference between the current model and check-pointed model. Using this method, the need to store old data is eliminated while ensuring that the model does not ‘forget’ the old data.


<p align="center">
  <img src="images/fig2_cruise.png" alt="PC grad idea" width="600" />
  <em>Fig. 2: The idea of PC grad</em>
</p>

I believe that the presentation fits well in the context of scalable AI. The authors not only introduce an important issue when dealing with scalable autonomous driving systems, but present interesting approaches towards solving it. The idea of generative AI being used to generate training samples in the context of robotics is gaining traction. It is being used in CNNs, path planning, RL-based control systems, and as already mentioned, in autonomous driving systems. This is important because collecting data for robotic systems is often a laborious task and consists of data from multiple sensors. Thus, collecting real-world data for unlikely scenarios may not be feasible for many applications. If generative AI can help in generating realistic data that is well diversified, the time-to-market can be significantly improved while ensuring that the model can generalize to complicated scenarios. However, a quantitative quality-of-sample should be established for applicability of the proposed method in real-world scenarios.

Further, in the context of continual learning, elimination of dependence on large amounts of data is essential in the longer term, however, the talk does go into a lot of detail regarding the performance benefits of PC grad when compared to other approaches. The idea is enticing and I believe that further analysis should be done to conclude the efficacy of the proposed method.

Though Dr. Chai addresses important issues towards scalable autonomous driving, additional research is necessary to measure and ensure the diversity of samples generated. Generative AI is difficult to train and can lead to inaccurate samples that might misguide the learning process. A metric to measure the ‘validity’ of generated samples should be devised which could help in discarding invalid samples. Such a metric would enable researchers to analyze the distribution of output generated by the AI and further tweak the AI to give more evenly distributed samples that constitute a varied set of scenarios.



<ol>
  <li>
    Samani, Ekta U., Feng Tao, Harshavardhan R. Dasari, Sihao Ding, and Ashis G. Banerjee. "F2BEV: Bird's Eye View Generation from Surround-View Fisheye Camera Images for Automated Driving." arXiv preprint arXiv:2303.03651 (2023).
  </li>
  <li>
    Talk 1: Scaling AV Across the US by Dr. Yuning Chai. 
  </li>
</ol>

