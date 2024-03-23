python --version
echo "Build"
pip3 install -r requirements.txt
pip3 install opencv-python
pip3 install opencv-contrib-python

echo "Python Version:"
python3.10 --version

echo "Installed Packages:"
pip3 freeze
