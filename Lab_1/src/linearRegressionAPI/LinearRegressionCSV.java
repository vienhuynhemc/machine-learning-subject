package linearRegressionAPI;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class LinearRegressionCSV {

	// Khai báo các thuộc tính
	// Dataset
	private Instances datasetTrain;
	private Instances datasetTest;

	public LinearRegressionCSV(String sourceTrain, String sourceTest) {
		// Nạp file
		loadFile(sourceTrain, sourceTest);
	}

	public LinearRegression getLinearRegressionModel() {
		LinearRegression linearRegressionModel = new LinearRegression();
		try {
			linearRegressionModel.buildClassifier(datasetTrain);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return linearRegressionModel;
	}

	public String evaluationResult(LinearRegression linearRegressionModel) {
		Evaluation evaluation = null;
		try {
			// Truyên vào dữ liệu chính là tập train
			evaluation = new Evaluation(datasetTrain);
			// Truyền vào
			// 1. Model
			// 2. Tập dữ liệu test
			// 3. Số dòng được test trong file test, ở đây chọn full
			// 4. Biến random
			evaluation.crossValidateModel(linearRegressionModel, datasetTest, datasetTest.numInstances(),
					new Random(1));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return evaluation.toSummaryString();
	}

	private void loadFile(String sourceTrain, String sourceTest) {
		CSVLoader csvLoader = new CSVLoader();
		try {
			// Train file
			csvLoader.setSource(new File(sourceTrain));
			datasetTrain = csvLoader.getDataSet();
			// Test file
			csvLoader = new CSVLoader();
			csvLoader.setSource(new File(sourceTest));
			datasetTest = csvLoader.getDataSet();
			// Set thuộc tính sẽ đc hồi quy là thuộc tính cuối cùng (quality)
			datasetTrain.setClassIndex(datasetTrain.numAttributes() - 1);
			datasetTest.setClassIndex(datasetTest.numAttributes() - 1);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("Không tìm thấy file");
		}
	}

}
