.. -*- mode: rst -*-

MDS-network
===========

.. image:: MDS-net.png

Multi-delay sinc (MDS) network for predicting continuous emotion annotations from speech signal

What is this repository?
------------------------

This repository contains a python project that can be used to replicate the experiments of the paper [1]. This paper introduces a new convolutional neural network (multi-delay sinc network). The network is able to simultaneously align and predict labels in an end-to-end manner. As it can be seen in the above figure, the network is a stack of convolutional layers followed by an aligner sub-network that aligns the speech signal and emotion labels. This aligner sub-network is implemented using a new convolutional layer that we introduce, the delayed sinc layer. It is a time-shifted low-pass (sinc) filter that uses a gradient-based algorithm to learn a single delay. Multiple delayed sinc layers can be used to compensate for a non-stationary delay that is a function of the acoustic space. 

How to run it?
--------------

To use this code and replicate an experiment of [1], three modules must be written: (1) data provider, (2) model, (3) run file.

* Data provider: this class manages the data usage of other parts of the code. There are some examples of data providers in the data_provider folder. The one that has been used in the experiments of the paper [1] is 'avec2016_provider.py'. I suggest to change this file and make it compatible with your dataset format. Each data provider class (such as the Avec2016Provider class) must inheret from the DataProvider class and implement following 4 functions:

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

* Model: 

Each run file is provided to replicate one experiment of the paper. Please feel free to contact me (Soheil Khorram), if you have any question regarding the current implementation.

References
----------

.. [1] Soheil Khorram, Melvin McInnis, Emily Mower Provost,
       *"Jointly Aligning and Predicting Continuous Emotion Annotations"*,
       IEEE Transactions on Affective Computing, 2019. [`PDF <https://arxiv.org/pdf/1907.03050.pdf>`_]

Author
------

- Soheil Khorram, 2019

