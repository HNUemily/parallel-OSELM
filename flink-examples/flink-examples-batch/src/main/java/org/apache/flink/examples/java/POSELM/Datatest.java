package org.apache.flink.examples.java.POSELM;

/**
 * Created by Administrator on 2017/4/25.
 */
import java.lang.*;
import java.io.*;



public class Datatest {

	public static void main(String[] args) {
		String s = "0.00000000  0.17647100 0.71000000 0.65573800 0.15000000 0.00000000 0.48286100 0.05209220 0.70000000";
		String[] tokens = s.split(" ");
		double [] array=new double[tokens.length];
		int i=0;
		for (String token:tokens)
		{
			array[i++]=Double.parseDouble(token);
		}
		for (i=0;i<tokens.length;i++)
		System.out.println(array[i]);
	}
}
