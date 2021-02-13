starttime=`date +'%Y-%m-%d %H:%M:%S'`
python prioritization.py cifar10_vgg16_example_test GF
python prioritization.py cifar10_vgg16_example_test WS
python prioritization.py cifar10_vgg16_example_test NAI
python prioritization.py cifar10_vgg16_example_test NEB
python model_feature_extraction.py cifar10_vgg16_example_test GF
python model_feature_extraction.py cifar10_vgg16_example_test WS
python model_feature_extraction.py cifar10_vgg16_example_test NAI
python model_feature_extraction.py cifar10_vgg16_example_test NEB
python feature_csv_conclusion.py cifar10_vgg16_example_test model
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "total_time:"$((end_seconds-start_seconds))"s"