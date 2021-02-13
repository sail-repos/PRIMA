# PRIMA
This is the implementation repository of our ICSE 2021 paper: Prioritizing Test Inputs for Deep Neural Networks via Mutation Analysis.
## Description

To relieve the labeling-cost problem, we propose a novel test input prioritization approach for DNNs via intelligent mutation analysis, called **PRIMA**.
PRIMA considers both model mutation and input mutation based on the following two key insights:
1) If a test input can kill many mutated models (by slightly changing the model under test), indicating that the test input can test the model sufficiently, the test input is likely to reveal DNN bugs.
2) If many mutated test inputs from one test input (by slightly changing the test input) have different prediction results with the original one on the model under test, indicating that much information of the test input is effectively utilized by the model, the test input is sensitive to capture DNN bugs.
They reflect the exploration degree of the test input to the DNN model under test and the test input itself, respectively.
Based on the key insights, PRIMA consists of three steps:
1) we design a series of model mutation rules and input mutation rules, and PRIMA obtains mutation results for each test input ;
2) PRIMA extracts a set of features from these mutation results for each test input in order to effectively utilize these mutation results to prioritize test inputs,
3) PRIMA adopts the framework of learning-to-rank to build a ranking model, which is able to intelligently utilize the extracted features, for prioritizing test inputs,

## Reproducibility

### Environments

We strongly recommend using package above (especially using tensorflow-gpu)

```tx
tensorflow==2.2.0
tensorflow-estimator==2.2.0
xgboost==1.0.1
Keras==2.3.1
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
pandas==0.25.3
Pillow==7.0.0
```



### Designed Features

| type  | name                          | domain |
| ----- | ------------------------------| ------ |
| Model | Gauss  Fuzzing                | /      |
|       | Neuron Effect Block           | /      |
|       | Neuron Activation Inverse     | /      |
|       | Weights Shuffling             | /      |
| Input | Pixel Gauss Fuzzing           | image  |
|       | Pixel Color Reverse           | image  |
|       | Coloring Pixel Black          | image  |
|       | Coloring Pixel White          | image  |
|       | Pixel Shuffling               | image  |
|       | Character Shuffling           | nlp    |
|       | Character Replacement         | nlp    |
|       | Character Repetition          | nlp    |
|       | Discrete Value Replacement    | form   |
|       | Continuous Value Modification | form   |



### Mutation

PRIMA contains 2 parts, 'input mutation' part, and 'model mutation' part. The results of both parts are features used for training in XGBOOST base learner.

#### input mutation

In the prioritization directory, we've got multiple files for feature extraction. We strongly recommend you using input_mutation_extraction.sh and model_mutation_extraction.sh for feature extraction.

datautils.py is the files where contains the path of your dataset and model, and necessary preprocessing procedures of  your dataset. You can modify the path and the name of your dataset (or model) in this file, then you can using input_mutation_extraction.sh or model_mutation_extraction.sh.

For input_mutation_extraction.sh

we've got multiple instructions in  input_mutation_extraction.sh.  input_mutation_extraction.sh is really flexible, in this case, we only provided a version of example so that you can using it fluently. 

```shell
python select_area_perturbated_generator.py exp_id mutation_rule
```

select_area_perturbated_generator.py is used to acquire input mutants for image classification.

$exp\_id$ is the name of your model-dataset pair, which can be set in datautils.py,  we've got 5 mutation_rule for image domain input, gauss stands for $Pixel\ Gauss\ Fuzzing$, shuffle stands for $Pixel\ Shuffling$, reverse stands for $Pixel\ Color\ Reverse$, white stands for $Coloring\ Pixel\ White$, black stands for $Coloring\ Pixel\ Black$. If you want to mutate input for natural language input or form type input, please use perturbate_nlp.py and perturbate_form.py, and the instructions are similar. 

Please note that you can modify the areas and pixels changing methods in select_area_perturbated_generator.py according to your requirements. In most of our subjects, the size of the image is 32*32 , so we provide you with the input mutation parameters according to this regular size. If you have subject with  larger image scale, you can change the setting parameters to fulfill your requirements. 

After generating input mutants, you can use

```shell
python acquire_prob.py exp_id mutation_rule
python feature_extraction.py exp_id mutation_rule
```

acquire_prob.py is used for predicting the probability vectors for feature extraction. test_case_num is the quantity of your dataset, we really recommend you using this parameter to guarantee this efficiency of mutation procedure. Please make sure that the exp_id, mutation_rule and test_case_num should be of consistency.

After you finish extract all input mutation features, please use feature_csv_conclusion.py to get the csv file of extracted features. 

```shell
python feature_csv_conclusion.csv exp_id domain
```

Please note that, the parameter domain has 4 choices, which is 'input','nlp','form' and 'model'. In your input mutation process, if your model-dataset pair belongs to image domain, please use 'input', if your task is in natural language domain, please use 'nlp'; if your task is in form (predefined feature)  domain, please use 'form'. Besides, if you are doing model mutation process, please use 'model'. Because each domain has its unique features and mutation rules. 

#### model mutation

Please note that, model mutation requires modifying the weights of the model high efficiently.  You can use provided mutation rules 'GF','NEB','NAI','WS' to get the mutants you want. This process can be implemented on your own or using related model mutation tools.

Please use  generate model mutants **BEFORE** running model_mutation_extraction.py.

### Feature extraction

#### input feature

After generating input mutants, you can use

```shell
python acquire_prob.py exp_id mutation_rule
python feature_extraction.py exp_id mutation_rule
```

acquire_prob.py is used for predicting the probability vectors for feature extraction. test_case_num is the quantity of your dataset, we really recommend you using this parameter to guarantee this efficiency of mutation procedure. Please make sure that the exp_id, mutation_rule and test_case_num should be of consistency.

After you finish extract all input mutation features, please use feature_csv_conclusion.py to get the csv file of extracted features. 

```shell
python feature_csv_conclusion.csv exp_id domain
```

Please note that, the parameter domain has 4 choices, which is 'input','nlp','form' and 'model'. In your input mutation process, if your model-dataset pair belongs to image domain, please use 'input', if your task is in natural language domain, please use 'nlp'; if your task is in form (predefined feature)  domain, please use 'form'. Besides, if you are doing model mutation process, please use 'model'. Because each domain has its unique features and mutation rules.

PLEASE note that, for convenience, we put the input mutants generation part into input_mutation_extraction.sh together with the extraction part because these procedures are connected tightly. 

#### model feature

After generating model mutants, you can use input_mutation_extraction.sh to acquire model mutants.

```shell
python prioritization.py exp_id mutation_rule
python model_feature_extraction.py exp_id mutation_rule
```

For model mutation,  the mutation rules are  'GF','NEB','NAI','WS'

'GF' 'NEB','NAI','WS' stands for $Gauss\ Fuzzing$ , $Neuron\ Effect\ Block$, $Neuron\ Activation\ Inverse$, $Weights\ Shuffling$ respectively. 

```shell
python feature_csv_conclusion.py exp_id domain
```

please use 'model' as the domain parameter to get model mutants features. It is generally used to get all types of model mutants (image,nlp, and predefined feature).



After this process , we can use the extracted features to train learn-to-rank model using XGBoost.

### Learning-to-Rank using XGBoost

After acquiring the extracted features, we can use it to build learning-to-rank model.

In ltr directory, xgboost_train.py provides you with the methods to build the learning2rank model. feature_utils.py is used to load test and validation features as well as other basic elements needed in building models.

After modifying the path of features in feature_utils.py , you can directly run xgboost_train.py to tune the parameters. You can change the state_num and n_estimator(epochs) to find the parameters that reaches highest effectiveness. We recommend you using the existing parameters, because the result is relatively stable and uniformly good.

xgboost_train.py can also save the ranking dict and the importance. We have already presented them in our   essay. We would like to contribute the extracted features in the supplement materials to assist analyzing.





## Dataset 

We conducted PRIMA on data-model above. Most of these data can be downloaded from their own page.
More specifically, we collected 

**8 image datasets**

- CIFAR-10 (a 10-class ubiquitous object dataset) [1]
- CIFAR-100 (a 100-class ubiquitous object dataset) [1]
- MNIST (a handwritten digit dataset) [2]
- MNIST\_VS\_USPS (a handwritten digit dataset for transfer learning) [3]
- COIL (a 20-class object recognition dataset for transfer learning [3]
- PIE27\_VS\_PIE5 (a face dataset for transfer learning) [3]
- PIE27\_VS\_PIE9 (a face dataset for transfer learning) [3]
- Driving (an autonomous driving dataset provided by Udacity) [4]

**5 text datasets**

- TREC (a question classification dataset) [5]
- IMDB (a large movie review dataset for binary sentiment classification) [6]
- SMS Spam (a mobile phone spam messages dataset) [7]
- CoLA (a linguistic acceptability dataset) [8]
- Hate Speech (a hate speech and offensive language collection dataset) [9]

 **1 dataset with predefined features**

- KDDCUP99 (a network intrusion information dataset provided by a competition in KDD'99).

You can easily download and use these data from the link down below.

![subjects](.\result\subjects.jpg)

## Additional Result

Due to the  limited space of the article, we presented the entire result here in result directory.

![subjects](.\result\rauc_all.jpg)



[1] CIFAR http://www.cs.toronto.edu/~kriz/cifar.html

[2] MNIST http://yann.lecun.com/exdb/mnist/

[3] TRANSFER LEARNING https://github.com/jindongwang/transferlearning/tree/master/data

[4] Driving https://udacity.com/self-driving-car

[5] TREC https://cogcomp.seas.upenn.edu/Data/QA/QC/

[6] IMDB http://ai.stanford.edu/~amaas/data/sentiment/

[7] SMS Spam http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

[8] CoLA https://nyu-mll.github.io/CoLA/

[9] Hate Speech https://github.com/t-davidson/hate-speech-and-offensive-language

[10] KDDCUP99 http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
