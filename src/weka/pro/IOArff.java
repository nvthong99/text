package weka.pro;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * @author Thong
 *
 */
public class IOArff {
	DataSource source;
	Instances dataset;
	Instances trainset;
	Instances testset;
	String[] model_option;
	String[] data_option;
	public IOArff() {
	
	}
	
	public IOArff(String filename) throws Exception {
		this.source = new DataSource(filename);
		this.dataset = source.getDataSet();
	}
	
	public void saveData(String filename) throws IOException {
		ArffSaver outData = new ArffSaver();
		outData.setInstances(this.dataset);
		outData.setFile(new File(filename));
		outData.writeBatch();
		System.out.println("Finished");
	}
	
	public Instances divideTraniTest( Instances originalSet, double percent, boolean isTest) throws Exception {
		RemovePercentage rp = new RemovePercentage();
		rp.setPercentage(percent);
		rp.setInvertSelection(isTest);
		rp.setInputFormat(originalSet);
		return Filter.useFilter(originalSet, rp);
	}
	
	public void setTrainset(String filename) throws Exception {
		DataSource trainSource = new DataSource(filename);
		this.trainset =  trainSource.getDataSet();
	}
	public void setTestset(String filename) throws Exception{
		DataSource testSource = new DataSource(filename);
		this.testset =  testSource.getDataSet();
	}
//	@Override
//	public String toString() {
//		// TODO Auto-generated method stub
//		return dataset.toSummaryString();
//	}
	
	
}
