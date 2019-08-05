.. -*- mode: rst -*-

MDS-network
===========

.. image:: MDS-net.png

Multi-delay sinc (MDS) network for predicting continuous emotion annotations from speech signals

What is this repository?
------------------------

This repository contains a python project that can be used to replicate the experiments of the paper [1]. This paper introduces a new convolutional neural network (multi-delay sinc network). The network is able to simultaneously align and predict labels in an end-to-end manner. As it can be seen in the above figure, the network is a stack of convolutional layers followed by an aligner sub-network that aligns the speech signal and emotion labels. This aligner sub-network is implemented using a new convolutional layer that we introduce, the delayed sinc layer. It is a time-shifted low-pass (sinc) filter that uses a gradient-based algorithm to learn a single delay. Multiple delayed sinc layers can be used to compensate for a non-stationary delay that is a function of the acoustic space. 

How to run it?
--------------

To use this code and replicate an experiment of [1], three modules must be written: (1) data provider, (2) model, (3) run file.

* Data provider: this class manages the data usage of other parts of the code. There are some examples of data providers in the data_provider folder. The one that has been used in the experiments of the paper [1] is 'avec2016_provider.py'. I suggest to change this file and make it compatible with your dataset format. Each data provider class (such as the Avec2016Provider class) must inherit from the DataProvider class and implement the following 4 functions:

.. code-block:: python

    def load_tr(self, opts):
        """Loads the train data and returns the number of train samples."""
        pass

    def load_dev(self, opts):
        """Loads the dev data and returns the number of development samples."""
        pass

    def load_te(self, opts):
        """Loads the test data and returns the number of test samples."""
        pass

    def get_sample(self, i):
        """Returns the i-th sample in the form of (utts, x, y)."""
        pass

* Model: This repository supports for TensorFlow models. I have prepared some examples in the 'models/tensorflow' folder. 'delay_attention.py' contains the MDS-network model shown in the above figure. Most experiments of the paper [1] use the model provided by 'delay_attention.py'. As the baseline system, we implemented the downsampling/upsampling network which is provided in 'conv_deconv.py'. Also, the preliminary experiment, presented in section 4 of the paper [1], uses the model exists in 'preliminary_cnn.py'. 

You can also write your own model. To do so, you just need to write a class that inherits from the 'TensorflowModel' and implements the following functions:

.. code-block:: python

    def construct(self):
        """Constructs a model."""
        pass

    @staticmethod
    def get_test_metrics():
        """Returns a list containing all test metrics."""
        pass

    @staticmethod
    def get_selection_metric():
        """Returns a metric for selecting best model."""
        pass

* Run file: in the run file, you define what data provider and what model you are going to use. You also define the parameters of the network and the parameters of the training procedure. There are many examples of the run files; Some important parameters are as follows:

    - model: is a variable that defines the type of the model (e.g., delay_attention, conv_deconv, ...).
    - data_provider: defines the data_provider (e.g., avec2016_provider).
    - exp-dir: results will be saved in this directory.
    - task: it is a parameter of the avec2016_provider data provider. It can be arousal or valence.
    - nb-epochs: number of training epochs.
    - lr: learning rate.
    - conv-kernel-len, conv-channel-num, conv-layer-num, sigma, delay-num, conv-l2-reg-weight, kernel-type: these some important parameters of the delay_attention model. For example, kernel-type can be gaussian or sinc and it defines the shape of the kernel used in the aligner sub-network of the MDS_network [1].
    
I have been provided many examples in these run files. Each of them is provided to replicate one experiment of the paper [1].

References
----------

.. [1] Soheil Khorram, Melvin McInnis, Emily Mower Provost,
       *"Jointly Aligning and Predicting Continuous Emotion Annotations"*,
       IEEE Transactions on Affective Computing, 2019. [`PDF <https://arxiv.org/pdf/1907.03050.pdf>`_]

Author
------

- Soheil Khorram, 2019

