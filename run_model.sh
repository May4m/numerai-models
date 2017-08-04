if [ "$(ls ./dump)" ]; then echo "Preparing visualization environment" && $(rm -rf ./dump/*); fi;
python convnet.py
