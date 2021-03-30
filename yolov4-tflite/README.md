
clone https://github.com/hhk7734/tensorflow-yolov4.git

py3.7

python setup.py install

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
