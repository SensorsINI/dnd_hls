unzip the train and test.zip
cd MLPTrainScripts
one needs to modify the dataset path in qtrain.py and qtest.py
trainfilepath = 'D://qmlpfpys-s//2xTrainingDataDND21train//'
testfilepath = 'D://qmlpfpys-s//2xTrainingDataDND21test//'

--for test the latest model
python qtestaccelerate.py dri 62000
--dri, application, could be 'hotel'
--62000, number of batches for testing dvs event records, could use any number < 62000, unzip the quantized-mlpf-runrocdata.rar and driytrue.rar and put the npy files under the same folder with the qtestaccelerate.py.


python qtrain.py 0 4
-first parameter 0/1, 0 means quantized training, 1 means float training.
-second parameter, 4 means quantized bits for inputs and activations before the last layer. the last layer have 16 bits.
-key code in function qbuildModel

python qtest.py hotel 4
-first parameter hotel/dri, hotel means using hotel dataset, dri means using driving dataset.
-second parameter, 4 means quantized bits.
-4bitweights0308.h5 can be used for wkfiles1 in qtest.py
