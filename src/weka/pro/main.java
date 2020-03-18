package weka.pro;

public class main {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		NaiveBayesModel model1 = new NaiveBayesModel();
		model1.buildNaiveBayesModel("train_test/train.arff");
		model1.evaluatetoNaivebayes("train_test/test.arff");
		
	//	model1.predictClassLabel("T:\\train_test\\iris_unlable.arff", "T:\\train_test\\iris_out.arff");
		
		SvmModel model = new SvmModel();
		model.buildSVM("train_test/train.arff");
		model.evaluatetoSVM("train_test/test.arff");
		
	}

}
