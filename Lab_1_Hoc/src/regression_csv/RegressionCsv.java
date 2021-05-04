package regression_csv;

import java.io.File;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class RegressionCsv {

	public static void main(String[] args) throws Exception {

		CSVLoader csvLoaderTrain = new CSVLoader();
		CSVLoader csvLoaderTest = new CSVLoader();
		csvLoaderTrain.setSource(new File("datasets/dataset1_train.csv"));
		csvLoaderTest.setSource(new File("datasets/dataset1_test.csv"));

		Instances dataTrain = csvLoaderTrain.getDataSet();
		dataTrain.setClassIndex(0);
		Instances dataTest = csvLoaderTest.getDataSet();

		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		dataTest.setClassIndex(dataTest.numAttributes() - 1);

		LinearRegression linearRegressionModel = new LinearRegression();
		linearRegressionModel.buildClassifier(dataTrain);
		System.out.println(linearRegressionModel);

		Evaluation evaluation = new Evaluation(dataTrain);
		evaluation.crossValidateModel(linearRegressionModel, dataTest, dataTest.numInstances(), new Random(1));
		System.out.println(evaluation.toSummaryString());

	}

}
