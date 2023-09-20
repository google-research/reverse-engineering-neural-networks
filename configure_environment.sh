conda create --name renn_env python=3.7
conda activate renn_env
conda install pip
pip install jax==0.1.70 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==0.1.50 -f https://storage.googleapis.com/jax-releases/jax_releases.html
conda install jupyter
conda install matplotlib
pip install tensorflow
pip install tensorflow-datasets==4.5.2
pip install tensorflow-text==2.2
pip install msgpack
pip install scikit-learn
pip install pyyaml
pip install ml-collections
pip install toolz
pip install jetplot
python setup.py install
