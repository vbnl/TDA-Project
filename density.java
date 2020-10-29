#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:28:16 2020

@author: nathanl
"""

import java.util.*;

public class density {

    //For each point, compute how large a radius is needed around it ...
    //... for a ball of such a radius around it to contain k points

    public ArrayList denseboi(int[] x, int[] y){

        /*EDIT AS NECESSARY*/
        //**********************************
        //radius parameter
        int r0 = 5;
        //minimum number of points allowable
        int k = 5;
        //**********************************

        //counter for points within r0
        int count = 0;

        //list of unacceptable points (by index)
        ArrayList<Integer> remove = new ArrayList<>();

        for (int i=0;i<x.length;i++){
            for (int j=0;j<x.length;j++){
                if (i != j){
                    double distance = Math.sqrt( (y[j] - y[i])*(y[j] - y[i])
                            + (x[j] - x[i])*(x[j] - x[i]) );
                    if (distance <= r0){
                        count+=1;
                    }
                }
                if (count > k){
                    remove.add(i);
                }
            }
        }
        return remove;
    }
}
