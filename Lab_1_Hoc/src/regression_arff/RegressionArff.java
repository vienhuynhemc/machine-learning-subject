package regression_arff;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RegressionArff {

	public static void main(String[] args) throws Exception {

		// Load datasource
		DataSource dataSourceTrain = new DataSource("datasets/dataset2_train.arff");
		DataSource dataSourceTest = new DataSource("datasets/dataset2_test.arff");
		Instances dataTrain = dataSourceTrain.getDataSet();
		Instances dataTest = dataSourceTest.getDataSet();

		// Xét lại biến cần tính hồi quy
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		dataTest.setClassIndex(dataTest.numAttributes() - 1);

		// Chạy linear regression
		LinearRegression linearRegressionModel = new LinearRegression();
		linearRegressionModel.buildClassifier(dataTrain);
		System.out.println(linearRegressionModel);

		// Test kết quả
		Evaluation evaluation = new Evaluation(dataTrain);
		evaluation.crossValidateModel(linearRegressionModel, dataTest, dataTest.numInstances(), new Random(1));
		System.out.println(evaluation.toSummaryString());

	}

}
