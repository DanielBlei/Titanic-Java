package titanic;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import weka.core.Instances;

//import weka.classifiers.Classifier;
//import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.meta.AdaBoostM1;
//import weka.classifiers.meta.Vote;
//import weka.classifiers.trees.J48;

import java.io.File;


public class Titanic{

    public static void main(String[] args)  throws Exception {

        //Converting the dataset to Arff

        String train_CSV = "/home/danielblei/Weka/files/train.csv";
        String train_ARFF = "/home/danielblei/Weka/files/train1.arff";
        String test_CSV = "/home/danielblei/Weka/files/test.csv";
        String test_ARFF = "/home/danielblei/Weka/files/test1.arff";

        Converter convert_train = new Converter(train_CSV,train_ARFF);
        Converter convert_test = new Converter(test_CSV,test_ARFF);

        //Loading the train and test set

        ArffLoader trainLoader = new ArffLoader();
        trainLoader.setSource(new File(train_ARFF));
        Instances train_Set = trainLoader.getDataSet();

        ArffLoader testLoader = new ArffLoader();
        testLoader.setSource(new File(test_ARFF));
        Instances test_Set = testLoader.getDataSet();

        //Removing unnecessary columns

        String[] option = new String[]{"-R", "1,4,8,9,11,12"};
        Remove remove = new Remove();
        remove.setOptions(option);
        remove.setInputFormat(train_Set);
        train_Set = Filter.useFilter(train_Set, remove);

        String[] option2 = new String[]{"-R", "1,3,7,8,10,11"};
        Remove remove2 = new Remove();
        remove2.setOptions(option2);
        remove2.setInputFormat(test_Set);
        test_Set = Filter.useFilter(test_Set, remove2);

        //Converting Sex type column to binary

        String[] nominal_to_binary = new String[]{"-R", "3"};
        NominalToBinary convertSex = new NominalToBinary();
        convertSex.setInputFormat(train_Set);
        convertSex.setOptions(nominal_to_binary);
        train_Set = Filter.useFilter(train_Set,convertSex);

        String[] nominal_to_binary2 = new String[]{"-R", "2"};
        NominalToBinary convertSex2 = new NominalToBinary();
        convertSex2.setInputFormat(test_Set);
        convertSex2.setOptions(nominal_to_binary2);
        test_Set = Filter.useFilter(test_Set,convertSex2);

        //Replacing the missing values from Fare attribute

        String[] replaceOpt = new String[]{"-R","6"};
        ReplaceMissingWithUserConstant replace = new ReplaceMissingWithUserConstant();
        replace.setOptions(replaceOpt);
        replace.setInputFormat(train_Set);
        train_Set = Filter.useFilter(train_Set,replace);

        String[] replaceOpt2 = new String[]{"-R","5"};
        ReplaceMissingWithUserConstant replace2 = new ReplaceMissingWithUserConstant();
        replace2.setOptions(replaceOpt2);
        replace2.setInputFormat(test_Set);
        test_Set = Filter.useFilter(test_Set,replace2);

        //Discretizing the Fare attribute in 10 bins

        String[] opt_factors = new String[5];
        opt_factors[0] = "-B";
        opt_factors[1] = "10"; // creating 10 bins
        opt_factors[2] = "-R";
        opt_factors[3] = "6";
        opt_factors[4] = "-F";
        Discretize discretize = new Discretize();
        discretize.setOptions(opt_factors);
        discretize.setInputFormat(train_Set);
        train_Set = Filter.useFilter(train_Set,discretize);

        String[] opt_factors2 = new String[5];
        opt_factors2[0] = "-B";
        opt_factors2[1] = "10"; // creating 10 bins
        opt_factors2[2] = "-R";
        opt_factors2[3] = "5";
        opt_factors2[4] = "-F";
        Discretize discretize2 = new Discretize();
        discretize2.setOptions(opt_factors2);
        discretize2.setInputFormat(test_Set);
        test_Set = Filter.useFilter(test_Set,discretize2);


        //Converting the Target attribute from the Train Set to binary

        String[] binary = new String[]{"-R","1,3"};
        NumericToBinary convertSurvived = new NumericToBinary();
        convertSurvived.setOptions(binary);
        convertSurvived.setInputFormat(train_Set);
        train_Set = Filter.useFilter(train_Set,convertSurvived);

        //Setting the index and the target variable

        train_Set.setClassIndex(train_Set.numAttributes()-1);

        Attribute trainAttribute = train_Set.attribute(0);
        train_Set.setClass(trainAttribute);


        //Building Model - Random Forest performed better results than the Vote Classifier

//        AdaBoostM1 Ada = new AdaBoostM1();
//        Ada.setNumIterations(250);
//
//        Classifier[] algorithms = {
//                new J48(),
//                Ada,
//                RF,
//                new NaiveBayes()
//        };
//        Vote average = new Vote();
//        average.setClassifiers(algorithms);

        RandomForest RF = new RandomForest();
        RF.setNumIterations(550);
        RF.buildClassifier(train_Set);

        //Evaluating the model performance
        Evaluation eval = new Evaluation(train_Set);

        eval.evaluateModel(RF, train_Set);

        System.out.println("=== Model Trained ===");
        System.out.println("Precision: "+eval.precision(1));
        System.out.println("Recall: "+eval.recall(1));
        System.out.println("F-Score: "+eval.fMeasure(1));
        System.out.println("AUC: "+eval.areaUnderROC(1));
        System.out.println("Kappa: "+eval.kappa());
        System.out.println("Accuracy: "+eval.pctCorrect()+"\n");
        System.out.println(eval.toMatrixString());

        //Saving the model
        SerializationHelper.write("titanic.model", RF);
        System.out.println("Trained model Saved");


        //Saving the Test Set containing the feature engineering alterations
        ArffSaver saver = new ArffSaver();
        saver.setInstances(test_Set); // data to convert
        saver.setFile(new File("/home/danielblei/Weka/files/test_edited.arff"));
        saver.writeBatch();
    }
}

