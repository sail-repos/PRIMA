starttime=`date +'%Y-%m-%d %H:%M:%S'`
python select_area_perturbated_generator.py cifar10_vgg16_example_test gauss
python select_area_perturbated_generator.py cifar10_vgg16_example_test reverse
python select_area_perturbated_generator.py cifar10_vgg16_example_test black
python select_area_perturbated_generator.py cifar10_vgg16_example_test white
python select_area_perturbated_generator.py cifar10_vgg16_example_test shuffle
python accquire_prob.py cifar10_vgg16_example_test gauss
python accquire_prob.py cifar10_vgg16_example_test reverse
python accquire_prob.py cifar10_vgg16_example_test black
python accquire_prob.py cifar10_vgg16_example_test white
python accquire_prob.py cifar10_vgg16_example_test shuffle
python feature_extraction.py cifar10_vgg16_example_test gauss
python feature_extraction.py cifar10_vgg16_example_test reverse
python feature_extraction.py cifar10_vgg16_example_test black
python feature_extraction.py cifar10_vgg16_example_test white
python feature_extraction.py cifar10_vgg16_example_test shuffle
python feature_csv_conclusion.csv cifar10_vgg16_example_test input
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "total_time:"$((end_seconds-start_seconds))"s"