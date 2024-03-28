**segfblla**
========

Installation
------------

::

        $ pip install .


Training
--------

To train on ALTO or Page XML files on a GPU:

::

        $ segfblla train -d cuda *.xml

Inference
---------

::

        $ segfblla seg -d cuda *.xml
