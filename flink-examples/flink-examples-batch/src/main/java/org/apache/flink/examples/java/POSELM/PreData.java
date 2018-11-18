package org.apache.flink.examples.java.POSELM;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class PreData {
	public static void main(String[] args){
		File inFile = new File("E:\\hnu\\DataForOSELM\\classifydata\\diabetes_train5"); // 读取的CSV文件
		File outFile = new File("E:\\hnu\\DataForOSELM\\classifydata\\diabetes_train10");//写出的CSV文件
		String inString = "";
		String tmpString = "";
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inFile));
			BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
			int LineNumber=0;
			while((inString = reader.readLine())!= null){
		        inString=LineNumber+" "+inString;
				writer.write(inString);
				writer.newLine();
				LineNumber++;
			}
			reader.close();
			writer.close();
		} catch (FileNotFoundException ex) {
			System.out.println("没找到文件！");
		} catch (IOException ex) {
			System.out.println("读写文件出错！");
		}
	}
}

