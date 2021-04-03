package main;

import linearRegressionAPI.LinearRegressionCSV;
import linearRegressionPerformedAgain.LinearRegessionPerformedAgain;
import weka.classifiers.functions.LinearRegression;

public class Main {

	public static void main(String[] args) {

		System.out.println("-------------------Linear Regression Api Weka--------------------");
		LinearRegressionCSV linearRegressionCSV = new LinearRegressionCSV("datasets/winequality-red_train.csv",
				"datasets/winequality-red_test.csv");
		// Lấy model regression và in ra
		LinearRegression model = linearRegressionCSV.getLinearRegressionModel();
		System.out.println(model);
		// Kiểm tra dữ liệu và in ra
		String evaluationResult = linearRegressionCSV.evaluationResult(model);
		System.out.println(evaluationResult);
		System.out.println("-----------------------------------------------------------------");
		System.out.println("---------------Linear Regression performed again-----------------");
		LinearRegessionPerformedAgain linearRegessionPerformedAgain = new LinearRegessionPerformedAgain(
				"datasets/winequality-red_train.csv", "datasets/winequality-red_test.csv", 0.001);
		// Chọn thuộc tính hồi quy là thuộc tính cuối cùng (quality)
		linearRegessionPerformedAgain.setClassIndex(linearRegessionPerformedAgain.numAttributes() - 1);
		// Chạy
		linearRegessionPerformedAgain.run();
	}

}
